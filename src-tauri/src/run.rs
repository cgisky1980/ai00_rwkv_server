use std::{
    borrow::Borrow,
    cmp::Ordering,
    collections::{HashMap, HashSet},
    convert::Infallible,
    sync::Arc,
    time::Instant,
};

use anyhow::Result;
use flume::{Receiver, Sender};
use itertools::Itertools;
use qp_trie::Trie;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use tokio::sync::{Mutex, RwLock};
use web_rwkv::{
    model::{v4, v5, v6, BackedState, FromBuilder, Model, ModelInfo, ModelState, StateBuilder},
    tokenizer::Tokenizer,
};

use crate::{Environment, FinishReason, GenerateRequest, Token, TokenCounter, STATE_CHUNK_SIZE};

const PENALTY_FREE_LIST: [&str; 5] = ["\n", ",", ".", "\u{002c}", "\u{002f}"];

#[derive(Debug)]
pub enum SlotResult {
    /// There is an idle slot ready to be picked up.
    Success(usize),
    /// An idle slot is swapped.
    Fault(usize),
    /// There is no idle slot left.
    Failure(Box<GenerateContext>),
    /// An error occurred.
    Error,
}

#[derive(Debug)]
enum SlotState {
    /// The slot might be either picked up or swapped.
    Idle(Tokens, Instant),
    /// The slot is locked and is waiting for processing.
    Wait(Box<GenerateContext>),
    /// The slot is currently under processing.
    Busy,
}

impl Default for SlotState {
    fn default() -> Self {
        Self::Idle(Default::default(), Instant::now())
    }
}

#[derive(Debug, PartialEq, Eq)]
enum SlotChoice {
    Continue(usize, usize),
    Back(usize),
    Empty(usize),
}

impl std::cmp::Ord for SlotChoice {
    fn cmp(&self, other: &Self) -> Ordering {
        // priority: continue > empty > back
        use SlotChoice::{Back, Continue, Empty};
        match (self, other) {
            (Continue(_, x), Continue(_, y)) => x.cmp(y),
            (Continue(_, _), _) => Ordering::Greater,
            (_, Continue(_, _)) => Ordering::Less,
            (Empty(_), Empty(_)) => Ordering::Equal,
            (Empty(_), Back(_)) => Ordering::Greater,
            (Back(_), Empty(_)) => Ordering::Less,
            (Back(_), Back(_)) => Ordering::Equal,
        }
    }
}

impl std::cmp::PartialOrd for SlotChoice {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Default)]
pub enum Payload {
    #[default]
    Empty,
    Busy(Box<GenerateContext>),
    Done(Box<GenerateContext>),
}

impl Payload {
    /// Takes out the value if `self` is [`Payload::Done`], and reset `self` to [`Payload::Empty`].
    pub fn take(&mut self) -> Option<Box<GenerateContext>> {
        match std::mem::take(self) {
            Payload::Done(context) => Some(context),
            payload => {
                *self = payload;
                None
            }
        }
    }

    /// Set `self` to [`Payload::Done`] if `self` is [`Payload::Busy`].
    pub fn finalize(&mut self) {
        *self = match std::mem::take(self) {
            Payload::Busy(context) => Payload::Done(context),
            payload => payload,
        }
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    pub fn is_busy(&self) -> bool {
        matches!(self, Self::Busy(_))
    }

    pub fn is_done(&self) -> bool {
        matches!(self, Self::Done(_))
    }
}

#[repr(transparent)]
#[derive(Debug, Default, Clone)]
pub struct Tokens(pub Vec<u16>);

impl std::ops::Deref for Tokens {
    type Target = TokenSlice;

    fn deref(&self) -> &Self::Target {
        self.0.as_token_slice()
    }
}

impl Borrow<[u8]> for Tokens {
    fn borrow(&self) -> &[u8] {
        bytemuck::cast_slice(&self.0)
    }
}

impl Borrow<[u16]> for Tokens {
    fn borrow(&self) -> &[u16] {
        &self.0
    }
}

impl Borrow<TokenSlice> for Tokens {
    fn borrow(&self) -> &TokenSlice {
        self.0[..].as_token_slice()
    }
}

impl qp_trie::Break for Tokens {
    type Split = TokenSlice;

    fn empty<'a>() -> &'a Self::Split {
        Default::default()
    }

    fn find_break(&self, loc: usize) -> &Self::Split {
        self.0[..loc >> 1].as_token_slice()
    }
}

#[repr(transparent)]
pub struct TokenSlice([u16]);

impl std::ops::Deref for TokenSlice {
    type Target = [u16];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Borrow<[u8]> for TokenSlice {
    fn borrow(&self) -> &[u8] {
        bytemuck::cast_slice(&self.0)
    }
}

impl Default for &TokenSlice {
    fn default() -> Self {
        <&[u16]>::default().as_token_slice()
    }
}

pub trait AsTokenSlice {
    fn as_token_slice(&self) -> &TokenSlice;
}

impl AsTokenSlice for [u16] {
    fn as_token_slice(&self) -> &TokenSlice {
        let ptr = self as *const [u16] as *const TokenSlice;
        unsafe { &*ptr }
    }
}

#[derive(Debug, Clone)]
pub struct GenerateContext {
    /// Tokens that are provided at first.
    pub prompt_tokens: Vec<u16>,
    /// Tokens that have been computed and cached.
    pub prefix: Tokens,
    /// Tokens to be computed.
    pub suffix: Tokens,
    /// The accumulated penalties for model-output tokens.
    pub penalties: HashMap<u16, f32>,
    /// Texts that are output by the model.
    pub model_text: Vec<u8>,
    /// Model may output partial utf-8. This makes sure the output is always valid.
    pub output_buffer: Vec<u8>,
    /// Tokens that are output by the model.
    pub model_tokens: Vec<u16>,
    /// Generate request provided by the caller.
    pub request: GenerateRequest,
    /// To send back generated tokens.
    pub sender: Sender<Token>,
}

#[derive(Debug, Clone)]
pub struct Runtime<M, S, B>
where
    B: BackedState,
    S: ModelState<BackedState = B>,
    M: Model<ModelState = S>,
{
    tokenizer: Arc<Tokenizer>,
    model: Arc<M>,
    state: Arc<S>,
    slots: Arc<Mutex<Vec<SlotState>>>,
    backed: Arc<Mutex<Trie<Tokens, B>>>,
    max_runtime_batch: usize,
    embed_layer: usize,
    penalty_free_tokens: HashSet<u16>,
}

impl<M, S, B> Runtime<M, S, B>
where
    for<'a> B: BackedState + Clone + FromBuilder<Builder<'a> = StateBuilder, Error = Infallible>,
    S: ModelState<BackedState = B>,
    M: Model<ModelState = S>,
{
    pub fn new(
        tokenizer: Tokenizer,
        model: M,
        state: S,
        max_runtime_batch: usize,
        embed_layer: usize,
    ) -> Self {
        let tokenizer = Arc::new(tokenizer);
        let model = Arc::new(model);
        let state = Arc::new(state);

        let slots = (0..state.max_batch())
            .map(|_| SlotState::default())
            .collect();
        let penalty_free_tokens = (0..u16::MAX)
            .filter(|&token| {
                let word = tokenizer.decode(&[token]).unwrap_or_default();
                let word = String::from_utf8_lossy(&word).into_owned();
                PENALTY_FREE_LIST.iter().any(|x| word.contains(x))
            })
            .collect();

        Self {
            tokenizer,
            model,
            state,
            slots: Arc::new(Mutex::new(slots)),
            backed: Arc::new(Mutex::new(Trie::new())),
            max_runtime_batch,
            embed_layer,
            penalty_free_tokens,
        }
    }

    pub fn info(&self) -> &ModelInfo {
        self.model.info()
    }

    pub fn tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }

    /// Queue an inference task.
    pub async fn queue(&self, context: GenerateContext) -> SlotResult {
        let mut slots = self.slots.lock().await;
        let mut cache = self.backed.lock().await;

        // we must ensure that there is at least one token as the suffix, otherwise the whole slot will loop forever as there is no input
        let (last, tokens) = match [context.prefix, context.suffix].concat().split_last() {
            Some((last, tokens)) => (*last, tokens.to_vec()),
            None => return SlotResult::Error,
        };

        // find the best idle slot by:
        // 1. find the slot that matches the context (continue)
        // 2. find an empty slot
        // 3. find the oldest non-empty slot
        let choice = slots
            .iter()
            .enumerate()
            .filter_map(|(batch, slot)| match slot {
                SlotState::Idle(content, time) => {
                    let delta = time.elapsed().as_millis();
                    match (content.is_empty(), tokens.starts_with(content)) {
                        (true, _) => Some((SlotChoice::Empty(batch), delta)),
                        (false, true) => Some((SlotChoice::Continue(batch, content.len()), delta)),
                        (false, false) => Some((SlotChoice::Back(batch), delta)),
                    }
                }
                _ => None,
            })
            .max_by(|lhs, rhs| lhs.0.cmp(&rhs.0).then(lhs.1.cmp(&rhs.1)));

        // here we try to search for the longest common prefix in the memory cache and checkout the state from that point
        // should there be a cache miss, an initial state is returned
        let mut checkout = |batch: usize| -> (Vec<u16>, B) {
            let prefix = cache.longest_common_prefix(tokens.as_token_slice());
            let len = (1..=prefix.len())
                .rev()
                .find(|len| cache.contains_key(prefix[0..*len].as_token_slice()))
                .unwrap_or_default();
            log::info!("slot {} checks out backed cache of length {}", batch, len);

            let prefix = prefix[0..len].to_vec();
            let reload = cache
                .remove(prefix[..].as_token_slice())
                .unwrap_or_else(|| {
                    let context = self.model.context();
                    let info = self.model.info();
                    StateBuilder::new(context, info)
                        .with_max_batch(1)
                        .with_chunk_size(STATE_CHUNK_SIZE)
                        .build_backed()
                });
            if len > 0 {
                let key = Tokens(prefix.clone());
                cache.insert(key, reload.clone());
            }
            (prefix, reload)
        };

        match choice {
            // we cannot find a slot because all slots are occupied
            // in this case, we hand the request back to the caller
            None => SlotResult::Failure(
                GenerateContext {
                    prefix: Default::default(),
                    suffix: Tokens([tokens, vec![last]].concat()),
                    ..context
                }
                .into(),
            ),
            // back a non-relative and non-empty slot and use it for our new context
            Some((SlotChoice::Back(batch), _)) => {
                log::info!("start at non-empty slot {}", batch);
                let (prefix, reload) = checkout(batch);

                let tokens = [tokens, vec![last]].concat();
                let len = prefix.len();
                let mut state = SlotState::Wait(
                    GenerateContext {
                        prefix: Tokens(tokens[..len].to_vec()),
                        suffix: Tokens(tokens[len..].to_vec()),
                        ..context
                    }
                    .into(),
                );

                std::mem::swap(&mut state, &mut slots[batch]);
                match state {
                    SlotState::Idle(content, _) => {
                        let backed = self.state.back_batch(batch).await.expect("back state");
                        cache.insert(content, backed);
                        self.state.load_batch(&reload, batch).expect("load state");
                        SlotResult::Fault(batch)
                    }
                    _ => unreachable!(),
                }
            }
            // directly occupy an empty slot so no need backing
            Some((SlotChoice::Empty(batch), _)) => {
                log::info!("start at empty slot {}", batch);
                let (prefix, reload) = checkout(batch);

                let tokens = [tokens, vec![last]].concat();
                let len = prefix.len();
                let state = SlotState::Wait(
                    GenerateContext {
                        prefix: Tokens(tokens[..len].to_vec()),
                        suffix: Tokens(tokens[len..].to_vec()),
                        ..context
                    }
                    .into(),
                );
                slots[batch] = state;

                self.state.load_batch(&reload, batch).expect("load state");
                SlotResult::Fault(batch)
            }
            // continue from an existing slot. No need backing as well
            Some((SlotChoice::Continue(batch, len), _)) => {
                log::info!("continue at slot {}", batch);
                let tokens = [tokens, vec![last]].concat();
                let state = SlotState::Wait(
                    GenerateContext {
                        prefix: Tokens(tokens[..len].to_vec()),
                        suffix: Tokens(tokens[len..].to_vec()),
                        ..context
                    }
                    .into(),
                );
                slots[batch] = state;
                SlotResult::Success(batch)
            }
        }
    }

    pub async fn process(&self, payloads: &mut Vec<Payload>) -> Result<()> {
        {
            let mut slots = self.slots.lock().await;
            let mut cache = self.backed.lock().await;

            payloads.resize_with(self.state.max_batch(), Default::default);
            // sync payloads and slots: kill dead payloads
            for (slot, payload) in slots.iter().zip_eq(payloads.iter_mut()) {
                if !payload.is_empty() && !matches!(slot, SlotState::Busy) {
                    *payload = Payload::Empty;
                }
            }

            // reset all finished slots to idle
            for (batch, payload) in payloads.iter_mut().enumerate() {
                if let Some(context) = payload.take() {
                    assert!(matches!(slots[batch], SlotState::Busy));
                    slots[batch] = SlotState::Idle(context.prefix, Instant::now());

                    if let Some(backed) = match &slots[batch] {
                        SlotState::Idle(content, _) => {
                            log::info!("backed slot {}", batch);
                            let backed = self.state.back_batch(batch).await.expect("back state");
                            cache.insert(content.clone(), backed.clone());
                            Some(backed)
                        }
                        _ => None,
                    } {
                        if context.request.embed {
                            let embed = backed.embed(0, self.embed_layer);
                            let _ = context.sender.send(Token::Embed(embed));
                        }
                    }
                }
            }

            // take data from some waiting slots
            let occupancy = payloads
                .iter()
                .filter(|x| matches!(x, Payload::Busy(_)))
                .count();
            let remain = self.max_runtime_batch - self.max_runtime_batch.min(occupancy);
            let batches = slots
                .iter()
                .enumerate()
                .filter(|(_, slot)| matches!(slot, SlotState::Wait(_)))
                .take(remain)
                .map(|(batch, _)| batch)
                .collect_vec();
            for batch in batches {
                let mut slot = SlotState::Busy;
                std::mem::swap(&mut slots[batch], &mut slot);
                match slot {
                    SlotState::Wait(context) => {
                        let _ = context.sender.send(Token::Start);
                        assert!(matches!(payloads[batch], Payload::Empty));
                        payloads[batch] = Payload::Busy(context);
                    }
                    _ => unreachable!(),
                };
            }
        };

        // shorter tasks first
        // let redirect = {
        //     let sorted = std::mem::take(payloads);
        //     let mut sorted = sorted.into_iter().enumerate().collect_vec();
        //     sorted.sort_unstable_by_key(|(_, payload)| match payload {
        //         Payload::Busy(context) => context.suffix.0.len(),
        //         _ => usize::MAX,
        //     });
        //     let (redirect, mut sorted): (Vec<_>, Vec<_>) = sorted.into_iter().unzip();
        //     std::mem::swap(payloads, &mut sorted);
        //     redirect
        // };

        let mut input_tokens = payloads
            .iter()
            .map(|payload| match payload {
                Payload::Busy(context) => context.suffix.0.clone(),
                _ => vec![],
            })
            .collect_vec();

        // run the model until there is at least one slot finished
        let occupancy = payloads.iter().filter(|x| x.is_busy()).count();
        let logits = match occupancy {
            0 => vec![None; payloads.len()],
            _ => loop {
                let logits = self.model.run(&mut input_tokens, &self.state).await?;
                if logits.iter().any(Option::is_some) {
                    break logits;
                }
            },
        };
        let penalty_free_tokens = &self.penalty_free_tokens;
        let logits = payloads
            .par_iter()
            .zip_eq(logits.into_par_iter())
            .map(|(payload, logits)| match payload {
                Payload::Busy(context) => logits.map(|mut logits| {
                    context
                        .penalties
                        .iter()
                        .filter(|(token, _)| !penalty_free_tokens.contains(token))
                        .for_each(|(token, penalty)| logits[*token as usize] -= penalty);
                    context
                        .request
                        .logit_bias
                        .iter()
                        .for_each(|(token, bias)| logits[*token as usize] += *bias);
                    logits
                }),
                _ => None,
            })
            .collect::<Vec<_>>();

        let probs = match occupancy {
            0 => vec![None; payloads.len()],
            _ => self.model.softmax(logits).await?,
        };
        let output_tokens = payloads
            .par_iter()
            .zip_eq(probs.into_par_iter())
            .map(|(payload, probs)| match payload {
                Payload::Busy(context) => probs.map(|probs| context.request.sampler.sample(probs)),
                _ => None,
            })
            .collect::<Vec<_>>();

        for (payload, token, tokens) in itertools::multizip((
            payloads.iter_mut(),
            output_tokens.into_iter(),
            input_tokens.into_iter(),
        )) {
            let mut done = false;

            if let Payload::Busy(context) = payload {
                let prefix = std::mem::take(&mut context.prefix);
                let suffix = std::mem::take(&mut context.suffix);
                let model_tokens = [prefix.0, suffix.0].concat();

                // compute new prefix and suffix using the current remaining tokens
                assert!(model_tokens.len() >= tokens.len());
                let len = model_tokens.len() - tokens.len();
                context.prefix = Tokens(model_tokens[..len].to_vec());
                context.suffix = Tokens(model_tokens[len..].to_vec());
                context
                    .penalties
                    .iter_mut()
                    .for_each(|(_, penalty)| *penalty *= context.request.sampler.penalty_decay);

                if let Some(token) = token {
                    assert_eq!(context.suffix.len(), 0);
                    context.suffix.0.push(token);

                    let penalty = match context.penalties.get(&token) {
                        Some(penalty) => penalty + context.request.sampler.frequency_penalty,
                        None => context.request.sampler.presence_penalty,
                    };
                    context.penalties.insert(token, penalty);

                    let mut word = self.tokenizer.decode(&[token])?;
                    context.model_text.append(&mut word.clone());
                    context.output_buffer.append(&mut word);
                    context.model_tokens.push(token);

                    // if let Ok(word) = String::from_utf8(context.output_buffer.clone()) {
                    //     let _ = context.sender.send(Token::Token(word));
                    //     context.output_buffer.clear();
                    // }

                    // let model_text = String::from_utf8_lossy(&context.model_text);
                    let count_tokens = || {
                        let prompt_tokens = context.prompt_tokens.len();
                        let completion_tokens = context.model_tokens.len();
                        let total_tokens = prompt_tokens + completion_tokens;
                        TokenCounter {
                            prompt_tokens,
                            completion_tokens,
                            total_tokens,
                        }
                    };
                    let mut finish = |reason| {
                        let _ = context.sender.send(Token::Stop(reason, count_tokens()));
                        let _ = context.sender.send(Token::Done);
                        done = true;
                    };

                    // let max_stop_len = context
                    //     .request
                    //     .stop
                    //     .iter()
                    //     .map(|stop| stop.len())
                    //     .max()
                    //     .unwrap_or_default()
                    //     .next_power_of_two();

                    let (output_pointer, stop_matched) = context
                        .request
                        .stop
                        .iter()
                        .map(|stop| {
                            let stop = stop.as_bytes();
                            let mut pointer_safe = 0;
                            let mut pointer_unsafe = 0;
                            while pointer_unsafe < context.output_buffer.len() {
                                // the maximum match of the current stop string
                                let pointer_stop = pointer_unsafe - pointer_safe;
                                if pointer_stop >= stop.len() {
                                    // we have a total match
                                    return (pointer_safe, true);
                                }

                                let output = context.output_buffer[pointer_unsafe];
                                let stop = stop[pointer_stop];

                                pointer_unsafe += 1;
                                if output != stop {
                                    pointer_safe = pointer_unsafe;
                                }
                            }
                            // end check
                            if pointer_unsafe - pointer_safe >= stop.len() {
                                (pointer_safe, true)
                            } else {
                                (pointer_safe, false)
                            }
                        })
                        .min_by(|x, y| match (x.1, y.1) {
                            (true, false) => Ordering::Less,
                            (false, true) => Ordering::Greater,
                            _ => x.0.cmp(&y.0),
                        })
                        .unwrap_or((context.output_buffer.len(), false));
                    let output = context.output_buffer[..output_pointer].to_vec();

                    if context.sender.is_disconnected() {
                        done = true;
                    } else if stop_matched {
                        let output = String::from_utf8_lossy(&output);
                        let _ = context.sender.send(Token::Token(output.into()));
                        finish(FinishReason::Stop)
                    } else if context.model_tokens.len() >= context.request.max_tokens {
                        finish(FinishReason::Length);
                    } else if let Ok(word) = String::from_utf8(output) {
                        let _ = context.sender.send(Token::Token(word));
                        context.output_buffer = context.output_buffer[output_pointer..].to_vec();
                    }
                }
            }

            if done {
                payload.finalize();
            }
        }

        // recover payload orders
        // let mut recovers = std::mem::take(payloads);
        // payloads.resize_with(payloads.len(), || Payload::Empty);
        // for index in 0..payloads.len() {
        //     std::mem::swap(&mut recovers[index], &mut payloads[redirect[index]]);
        // }

        Ok(())
    }
}

pub enum RuntimeUntyped<'a> {
    V4(Runtime<v4::Model<'a>, v4::ModelState, v4::BackedState>),
    V5(Runtime<v5::Model<'a>, v5::ModelState, v5::BackedState>),
    V6(Runtime<v6::Model<'a>, v6::ModelState, v6::BackedState>),
}

macro_rules! impl_runtime_untyped {
    ($($variant:ident),* $(,)?) => {
        impl RuntimeUntyped<'_> {
            #[inline]
            pub fn info(&self) -> &ModelInfo {
                match self {
                    $(RuntimeUntyped::$variant(runtime) => runtime.info(),)*
                }
            }

            #[inline]
            pub fn tokenizer(&self) -> Arc<Tokenizer> {
                match self {
                    $(RuntimeUntyped::$variant(runtime) => runtime.tokenizer(),)*
                }
            }

            #[inline]
            pub async fn queue(&self, context: GenerateContext) -> SlotResult {
                match self {
                    $(RuntimeUntyped::$variant(runtime) => runtime.queue(context).await,)*
                }
            }

            #[inline]
            pub async fn process(&self, payloads: &mut Vec<Payload>) -> Result<()> {
                match self {
                    $(RuntimeUntyped::$variant(runtime) => runtime.process(payloads).await,)*
                }
            }
        }
    };
}

impl_runtime_untyped!(V4, V5, V6);

#[tokio::main]
pub async fn run(receiver: Receiver<()>, env: Arc<RwLock<Environment<'_>>>) {
    while let Ok(()) = receiver.recv_async().await {
        let mut payloads = vec![];
        'run: loop {
            if let Environment::Loaded { runtime, .. } = &*env.read().await {
                if let Err(err) = runtime.process(&mut payloads).await {
                    log::error!("{}", err);
                }
            }
            if payloads.iter().all(Payload::is_empty) {
                break 'run;
            }
        }
    }
}
