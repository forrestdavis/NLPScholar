- [ ] Masked language modeling with a sliding window needs more work. 
    - [ ] Care with adding to each batch 
    - [ ] Refactor the handling of CLS and SEP, perhaps again we need something like 0/1 for both 
- [ ] Refactor to remove last_non_masked_idx 
- [ ] Test masked model with sliding window at edge of context length, the CLS/SEP addition has a two token mismatch between what I am viewing as context and what the model sees
- [ ] Refactor ppl calculator (or ensure it is still working) and rename as
  per-batch ppl 
