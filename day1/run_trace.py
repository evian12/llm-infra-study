import argparse

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Local Hugging Face model path or model ID",
    )
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        max_model_len=512,
        max_num_batched_tokens=32,
        max_num_seqs=4,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        gpu_memory_utilization=0.7,
    )

    # 构造不同长度的普通文本请求
    prompt_a = (
        "Explain why GPU batching is useful. "
        + "Explain the scheduling mechanism. " * 10
    )

    prompt_b = (
        "Describe how a transformer processes tokens. "
        + "Explain prefill and decode in detail. " * 22
    )

    prompts = [prompt_a, prompt_b]

    params = SamplingParams(
        temperature=0,
        max_tokens=5,
        ignore_eos=True,
    )

    print("Starting Day 1 Scheduler Trace", flush=True)

    for i, prompt in enumerate(prompts):
        print(
            f"request={i}, chars={len(prompt)}",
            flush=True,
        )

    outputs = llm.generate(
        prompts,
        sampling_params=params,
    )

    print("\n=== FINAL OUTPUTS ===")
    for i, output in enumerate(outputs):
        print(f"Request {i}:")
        print("prompt_tokens:", len(output.prompt_token_ids))
        print("generated_tokens:", len(output.outputs[0].token_ids))
        print("text:", output.outputs[0].text[:150])


if __name__ == "__main__":
    main()