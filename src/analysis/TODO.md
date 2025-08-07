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


## Token Classification