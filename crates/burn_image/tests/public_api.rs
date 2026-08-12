use burn_image::{Dimensions, GenerationOptions, Prompt};

#[test]
fn model_neutral_request_types_are_public_correctness() {
    let dimensions = Dimensions::new(1024, 1024).unwrap();
    let prompt = Prompt::new("a red cube").unwrap();
    let options = GenerationOptions {
        dimensions: Some(dimensions),
        seed: Some(42),
        ..GenerationOptions::default()
    };
    assert_eq!(prompt.as_str(), "a red cube");
    assert_eq!(options.dimensions, Some(dimensions));
    assert_eq!(options.seed, Some(42));
}
