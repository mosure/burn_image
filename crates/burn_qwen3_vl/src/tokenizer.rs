//! Optional adapter for Hugging Face `tokenizer.json` files.

use std::path::Path;

use crate::{Qwen3VlError, Result, processor::Qwen3VlTokenizer};

#[derive(Clone)]
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
}

impl core::fmt::Debug for HfTokenizer {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("HfTokenizer")
            .finish_non_exhaustive()
    }
}

impl HfTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|error| Qwen3VlError::Tokenizer(error.to_string()))?;
        Ok(Self { inner })
    }

    pub fn from_bytes(json: &[u8]) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_bytes(json)
            .map_err(|error| Qwen3VlError::Tokenizer(error.to_string()))?;
        Ok(Self { inner })
    }

    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}

impl Qwen3VlTokenizer for HfTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i64>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|error| Qwen3VlError::Tokenizer(error.to_string()))?;
        Ok(encoding.get_ids().iter().map(|&id| i64::from(id)).collect())
    }

    fn decode(&self, ids: &[i64], skip_special_tokens: bool) -> Result<String> {
        let ids = ids
            .iter()
            .map(|&id| {
                u32::try_from(id).map_err(|_| {
                    Qwen3VlError::Tokenizer(format!("token id {id} does not fit into u32"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.inner
            .decode(&ids, skip_special_tokens)
            .map_err(|error| Qwen3VlError::Tokenizer(error.to_string()))
    }

    fn token_to_id(&self, token: &str) -> Option<i64> {
        self.inner.token_to_id(token).map(i64::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_tokenizer_json_reference() {
        let Ok(path) = std::env::var("QWEN3_VL_TOKENIZER_JSON") else {
            return;
        };
        let tokenizer = HfTokenizer::from_file(path).unwrap();
        assert_eq!(tokenizer.token_to_id("<|im_start|>"), Some(151_644));
        assert_eq!(tokenizer.token_to_id("<|image_pad|>"), Some(151_655));
        assert_eq!(tokenizer.token_to_id("<|video_pad|>"), Some(151_656));
        let ids = tokenizer
            .encode("<|im_start|>user\nhello<|im_end|>\n", false)
            .unwrap();
        assert_eq!(ids.first(), Some(&151_644));
        assert!(ids.contains(&151_645));
    }
}
