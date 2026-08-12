use burn_qwen3_vl::Qwen3VlConfig;

#[test]
fn reusable_qwen_config_is_boogu_independent_correctness() {
    let config = Qwen3VlConfig::from_json(
        r#"{
            "text_config": {
                "vocab_size": 64, "hidden_size": 8, "intermediate_size": 16,
                "num_hidden_layers": 1, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 4,
                "rope_scaling": {"mrope_section": [2, 0, 0], "mrope_interleaved": true}
            },
            "vision_config": {
                "depth": 1, "hidden_size": 8, "intermediate_size": 16,
                "num_heads": 2, "patch_size": 2, "temporal_patch_size": 1,
                "spatial_merge_size": 2, "out_hidden_size": 8,
                "num_position_embeddings": 4, "deepstack_visual_indexes": [0]
            },
            "tie_word_embeddings": false,
            "image_token_id": 60, "video_token_id": 61,
            "vision_start_token_id": 62, "vision_end_token_id": 63
        }"#,
    )
    .unwrap();
    config.validate().unwrap();
    assert_eq!(config.text_config.hidden_size, 8);
}
