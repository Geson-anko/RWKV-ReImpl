"""This file contains the functions for generation."""
import torch


def rwkv_generate_fixed_length(
    net: torch.nn.Module,
    prompt_tokens: torch.Tensor,
    num_generating_tokens: int,
    prepare_hidden_state_recursively: bool = True,
) -> torch.Tensor:
    """Generate text of fixed length.

    Args:
        net: Model.
        prompt_tokens: 1D tokens array to start generation with. Device must be the same as net's.
        num_generating_tokens: Number of tokens to generate per sample.
        prepare_hidden_state_recursively: Whether to prepare hidden state recursively.
    
    Returns:
        generated_tokens: Generated tokens. Shape (num_generating_tokens,).
    """
    if prepare_hidden_state_recursively:
        for token in prompt_tokens:
            out = net(token.view(1,1)) # (len=1, batch=1, vocab_size)
    else:
        out = net(prompt_tokens.unsqueeze(1))[-1] # (batch=1, vocab_size)
    
    out = torch.argmax(out.squeeze(), dim=-1)
    generated_tokens = [out]
    for _ in range(num_generating_tokens-1):
        out = net(out.view(1,1))
        out = torch.argmax(out.squeeze(), dim=-1)
        generated_tokens.append(out)

    return torch.stack(generated_tokens)

