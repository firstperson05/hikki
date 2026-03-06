use std::collections::HashMap;

/// Vocabulary wrapping HashMap<String, u32> + Vec<String>
/// Special tokens: <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3
#[derive(Debug, Clone)]
pub struct Vocab {
    pub token_to_id: HashMap<String, u32>,
    pub id_to_token: Vec<String>,
}

impl Vocab {
    pub fn new() -> Self {
        let mut vocab = Vocab {
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
        };
        // Add special tokens in order
        vocab.add_token("<PAD>");
        vocab.add_token("<UNK>");
        vocab.add_token("<BOS>");
        vocab.add_token("<EOS>");
        vocab
    }

    pub fn add_token(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }
        let id = self.id_to_token.len() as u32;
        self.token_to_id.insert(token.to_string(), id);
        self.id_to_token.push(token.to_string());
        id
    }

    pub fn token_to_id(&self, token: &str) -> u32 {
        *self.token_to_id.get(token).unwrap_or(&1) // 1 is <UNK>
    }

    pub fn id_to_token(&self, id: u32) -> &str {
        if (id as usize) < self.id_to_token.len() {
            &self.id_to_token[id as usize]
        } else {
            "<UNK>"
        }
    }

    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_specials() {
        let v = Vocab::new();
        assert_eq!(v.len(), 4);
        assert_eq!(v.token_to_id("<PAD>"), 0);
        assert_eq!(v.token_to_id("<UNK>"), 1);
        assert_eq!(v.token_to_id("<BOS>"), 2);
        assert_eq!(v.token_to_id("<EOS>"), 3);
        assert_eq!(v.id_to_token(2), "<BOS>");
    }

    #[test]
    fn test_vocab_add() {
        let mut v = Vocab::new();
        let id = v.add_token("hello");
        assert_eq!(id, 4);
        assert_eq!(v.token_to_id("hello"), 4);
        assert_eq!(v.id_to_token(4), "hello");
    }
}
