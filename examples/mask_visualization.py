"""Example: Attention mask visualization."""
import torch
from apex.model.mask import build_apex_attention_mask

def main():
    print("=== APEX-1 Attention Mask Visualization ===\n")
    prefix_len, total_len, window = 4, 12, 4

    print("Global layer mask (full causal + prefix bidir):")
    g = build_apex_attention_mask(prefix_len, total_len, window, is_global_layer=True)
    _print_mask(g)

    print("\nLocal layer mask (sliding window + prefix bidir):")
    l = build_apex_attention_mask(prefix_len, total_len, window, is_global_layer=False)
    _print_mask(l)

    print("\nLegend: █=attend  ·=masked")
    print("Rows 0-3 are prefix (bidirectional), rows 4+ are generation (causal)")

def _print_mask(mask):
    for i in range(mask.shape[0]):
        row = ""
        for j in range(mask.shape[1]):
            row += "█ " if mask[i, j] else "· "
        print(f"  pos {i:2d}: {row}")

if __name__ == "__main__":
    main()
