# This project is inactive
I may pick it up again in the future, but probably not. My interests have moved away from NLP via neural networks, and there are a lot of challenges associated to the machine learning problem here at any rate.

# xword-net
A neural network for solving crossword puzzles. The current scope is limited to clues whose answer is exactly 5, as I haven't fully figured out how I want to handle training with variable-length output.

This kind of training is really only possible with a database of crossword clues, and luckily we have Matt Ginsberg's excellent resource - http://www.otsys.com/clue/. The file clues-5-stripped.txt was made from processing the clues.bz2 he offers.

You'll also need to download the word2vec model (1.5gb) from https://code.google.com/archive/p/word2vec/: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing, and extract it here. In the future, we may support or use in addition other pre-trained word2vec embeddings, see http://ahogrammer.com/2017/01/20/the-list-of-pretrained-word-embeddings/.
