## Minimal Pair

 - [x] Have a flag that outputs different interim files --> Maybe this can be phrased as what granularity of "analysis" does user want. FIX: added `save` flag


 - [x] Generalize setup of lemmas and contextids so that the aggregation happens over any column. FIX: added `condition` flag that takes any number of column names for condition. Also ignores invalid columns (but warns user. )


 - [x] Verify that every pair ID in the TSV file has expected and unexpected. If not, drop pairid and print that pairIDs were dropped. 

 - [x] Get rid of get_diff function. Always have differece be unexpected - expected (and state that the we want values to be negative for prob diff) 

 - [x] Make ROI specification optional. If no ROI, aggregate over the entire sentence. 

 - [x] Remove top-k and lemma filtering. Was too niche -- can be accomplished more easily by user if required from the interim files that are generated. 

 - [x] Get rid of micro vs. macro distinction for `roi_summary` -- easier to break (e.g., if ROIs don't have exact same number across tokens), and practically hardly any difference between micro and macro (as per simulation below). 
  ```
a = np.random.rand(1000)
b = np.random.rand(1000)

print((a-b).mean())
print(a.mean() - b.mean())

(a-b).mean() == a.mean() - b.mean()
 ```

 - [ ] Let comparison take manually specified labels (with default being expected and unexpected)


## Text Classification

- [x] Make MWE for text classification slightly more complex with condition (e.g., maybe have long/short, as well as source as conditions?)

- [x] Create different granularity of output types: by_cond, by_class 

- [x] If there is by_class make sure to also have a micro-average and macro-average option? 

- [] Compute metrics:
    - [x] accuracy
    - [x] precision
    - [x] recall
    - [x] f-beta 
    - [ ] Something to do with the probability of the prediction? Something like P(pred | correct) and P(pred | incorrect)

- [x] Verify that this works with both paired and not-paired sentences


## Token Classification
- [x] Get one label per word: aggregate strategies:
    - [x] take first token label and prob
    - [x] take max prob and get the token with that
    - [ ] take mode label and get mean prob of everything with mode label.   
- [ ] Add a ignore column. By default ignore is false for everything.
- [ ] Aggregrate outputs: by_cond, by_class (where ignore is filtered out)
- [ ] Make MWE for POS (where there is nothing to ignore) and NER (where we want to ignore the non-entity) more complex by adding in condition 
- [ ] Figure out how to specify id2label for POS tagging 


## Other

- [x] Move the condition stuff from MinimalPair to Analysis. Check it works
- [x] Remove condition and other things as a thing that evaluate needs. Possibly have a check in analysis?
- [ ] Make sample configs for each experiment and mode type. Possibly have subfolders for each experiment type

