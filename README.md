## About
This project is inspired from the work of *Mihalcea et al.* published in the paper, <a href = "https://www.aclweb.org/anthology/C04-1162.pdf"> *PageRank on Semantic Networks with Application to Word Sense Disambiguation* </a>. The task of word sense disambiguation consists of assigning the most appropriate meaning to a *polysemous* word within a given context. The paper presents an open text word sense disambiguation method that combines the use of logical inferences with *PageRank-style algorithms* applied on graphs extracted from natural language documents. *PageRank* algorithm is a way of measuring the importance of a web page (in a directed graph of web-pages) depending on the number of other pages linked to this page and the number of pages to which this page is linked. This algorithm is extremely extensible and can be used to rank any class of entities arranged in a graph-ic fashion. This intuition behind a *PageRank* algorithm can be used to develop a similar algorithm to measure the importance of a *word sense* in a huge network of senses.

*WORD_SENSE_DISAMBIGUATOR.py* realizes this approach given in the paper. Given a text file, *WORD_SENSE_DISAMBIGUATOR.py* will annotate those words and phrases in the text that have an ambiguous meaning. Following the annotated text, a table would present the most appropriate senses allocated to these ambiguous words/phrases, deduced by the proposed algorithm. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/66432513/120797224-5bedbb80-c559-11eb-8c35-f788074c9bb7.png" width = '900' height = '990'>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/66432513/120797226-5db77f00-c559-11eb-8485-120a963ac75b.png" width = '900' height = '990'>

## Graph Construction
The most important step before implementing a *PageRank*-style algorithm is to translate all the raw data into a graph. The *Text Synset Graph* constructed in this step has *word senses* as its nodes. Every word in the graph that has an ambiguous meaning must be having multiple senses. Even the unambiguous senses must have a unique sense associated with them. Post the *pre-processing* stage, the nodes in the network are identified as the senses of words (and also collocations) in the text. Note that given an ambiguous word, only those senses are chosen as candidates that have the same *part of speech tag* as that word in its context. For example, given the word *tick* in a sentence *"The comforting tick of..."*, the *verb* senses of this word would not be considered as candidates. In the pre-processing stage, *named entities* are identified (names of people, places, books etc.) because these are always unambiguous.

The next important task is introducing edges in the network. *WordNet* is a lexical knowledge base for English that defines words, meanings, and relations between them. The basic unit in WordNet is a *synset*, which is a set of synonym words or word phrases that represents a concept. *Synset* can be rightly considered as a *sense* that might be shared by multiple words in the vocabulary. The nodes in the *Text Synset Graph* are actually the *Synset* objects defined in Python's WordNet module. An undirected edge is introduced between two nodes if and only if the corresponding words/collocations, of which they are a candidate, are present within some context window of each other; and their corresponding synsets share some semantic relationship defined in the WordNet. *Colexical* synsets (the ones having the same root-name attribute, like *Synset('dog.n.01')* and *Synset('dog.n.03')* have the same root-name *dog*) are always isolated from each other. WordNet defines several semantic relations between synsets. The ones used in this project while constructing the graph are -- *Synonymy*, *Hypernymy*, *Hyponymy*, *Meronymy*, *Holonymy*, *Entailment*, *Coordinate* and *Exceptional* (exceptional relationships are not given in the paper but they are used to account for the pairs of synsets that do not share any of the other concrete relationships but show exceptionally high *WuPalmer* similarity, say above 75%).

## Graph Sparseness and Breaking Ties
The synset graphs generated in the first step are extremely *sparse*, even for texts with word count above just 500. In the network, the fraction of isolated nodes can be as high as 82% in these cases. This is reasonable because in the network, we are taking all (compatible) senses of almost all the words (even the unambiguous ones), excluding only the named entities and the words belonging to *parts of speech* other than *Noun*, *Adjective*, *Verb* and *Adverb* (that cover the majority of the tokens). Besides, senses of collocations (only those defined in the WordNet) are also included in the network. The semantic relationships used to connect two senses are very specific and most of the pairs will not belong to any class of the given relationships; and hence possibly many senses, that are not too commonly used in language, might get isolated.

The problem with sparseness is that all the isolated nodes will get the same score (equal to the *damping constant* in the *PageRank*-style algorithm) after the *PageRank*-style algorithm terminates. Breaking ties, in this case, becomes a crucial task. Breaking ties randomly might produce highly inconsistent results. Because of the extreme sparseness of the network, tie-breaking becomes a very important matter to disambiguate the text. A pretty effective tie-breaking strategy that is used is described in the following snippet (*Google*'s pre-trained *Word2Vec* model is used to get word-embeddings).

    def CheckSimilarityWithDefinition ( word_info , definition ) :
        sent_loc, word_loc, word = word_info
        tokens_def = GetRefinedTokens(definition)
        tokens_text = set([w.lower() for w, synsets in SENTENCE_TOKENS[sent_loc]])
        valid_pairs = 0
        net_similarity = 0.0
        for token_text in tokens_text :
            embed_txt = GetTokenEmbedding(token_text)
            if ( embed_txt[0] == 10**10 ) : continue
            for token_def in tokens_def :
                embed_def = GetTokenEmbedding(token_def)
                valid_pairs += 1
                net_similarity += np.linalg.norm(embed_def - embed_txt)
        if ( valid_pairs == 0 ) : return float('inf')
        ratio = net_similarity / valid_pairs
        relative_change_for_fractional_part = 0.0001 / round(ratio)
        if ( relative_change_for_fractional_part < 10**(-6) ) : return float('inf')
        return ratio

    def BreakTies ( word_info , sense1 , sense2 ) :
        sent_loc, word_loc, word = word_info
        name1 = GetSynsetName(sense1)
        name2 = GetSynsetName(sense2)
        if ( name1 == name2 ) :
            inv_similarity1 = CheckSimilarityWithDefinition(word_info, sense1.definition())
            inv_similarity2 = CheckSimilarityWithDefinition(word_info, sense2.definition())
            if ( inv_similarity1 < inv_similarity2 ) : return sense1
            return sense2
        embed1 = GetTokenEmbedding(word)
        embed2 = GetTokenEmbedding(name1)
        embed3 = GetTokenEmbedding(name2)
        inv_similar1 = np.linalg.norm(embed1 - embed2)
        inv_similar2 = np.linalg.norm(embed1 - embed3)
        if ( inv_similar1 < inv_similar2 ) : return sense1
        return sense2

