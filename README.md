## About
This project is inspired from the work of *Mihalcea et al.* published in the paper, <a href = "https://www.aclweb.org/anthology/C04-1162.pdf"> *PageRank on Semantic Networks with Application to Word Sense Disambiguation* </a>. The task of word sense disambiguation consists of assigning the most appropriate meaning to a *polysemous* word within a given context. The paper presents an open text word sense disambiguation method that combines the use of logical inferences with *PageRank-style algorithms* applied on graphs extracted from natural language documents. *PageRank* algorithm is a way of measuring the importance of a web page (in a directed graph of web-pages) depending on the number of other pages linked to this page and the number of pages to which this page is linked. This algorithm is extremely extensible and can be used to rank any class of entities arranged in a graph-ic fashion. This intuition behind a *PageRank* algorithm can be used to develop a similar algorithm to measure the importance of a *word sense* in a huge network of senses.

*WORD_SENSE_DISAMBIGUATOR.py* realizes this approach given in the paper. Given a text file, *WORD_SENSE_DISAMBIGUATOR.py* will annotate those words in the text that have an ambiguous meaning. Following the annotated text, a table would present the most appropriate senses allocated to these ambiguous words, deduced by the proposed algorithm.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/66432513/120797224-5bedbb80-c559-11eb-8c35-f788074c9bb7.png" width = '800' height = '900'>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/66432513/120797226-5db77f00-c559-11eb-8485-120a963ac75b.png" width = '800' height = '900'>
