## Applications of Deep Learning - Natural Language Understanding: Are Two Sentences of the same Topic

This is a project from the course  [T81-855: Applications of Deep Learning at Washington University in St. Louis](https://sites.wustl.edu/jeffheaton/t81-558/). In this competition you are given two sentences. You should determine the likelihood that these two sentences are from the same source.

# Sentences from Same Source

These two are describing demographics of a town:

-   The population density was 181.0 per square mile (69.9/km¬≤).
-   The racial makeup of the township was 96.91% White, 0.75% African American, 0.41% Native American, 0.14% Asian, 0.91% from other races, and 0.88% from two or more races.

Describing a monument:

-   Monument of the Judiciary Citadel of Salerno, near the Colle Bellara.
-   The La Carnale Castle got his name from a medieval battle against the Arabs and is part of a sport complex (with pool, tennis courts and hockey).

Describing a sport:

-   Mountain unicycling is an adventure sport that consists of traversing rough terrain on a unicycle.
-   The banana saddle is fixed (not spring suspended) on the fork and carries a front and back bumper made of stiff plastics.
# Sentences Not from Same Source

Not the same:

-   Meanwhile, Western foods which are rich in fat, salt, sugar, and refined starches are also imported into countries.
-   According to the United States Census Bureau, the CDP has a total area of , of which is land and (3.61%) is water.

Not the same:

-   The EIOC also is available to utilities, vendors, government agencies and universities interested in research, development or training.
-   Earlier that year, Environment secretary Michael Gove announced that a panel would be formed to consider making some of the AONBs into National Parks.

Not the same:

-   Due to financial concerns, it was announced in September 1968 that the target date for completing the Interstate Highway System would be extended until 1974, from 1972.
-   In doing so, it formed a networking point for zine creators and readers (usually the same people).

# Steps:
- Pre-processing Text Data: Tokenization and so on
- Feature Engineering: 
	 1. uncommon words in the sentence
	 2. the number of words appearing on both sentences
	 3. the number of time words
	 4. the number of same words
	 5. topic modeling
	 6. part of speech
	 7. ...
- Deep Learning Model (xgboost, kfold)
