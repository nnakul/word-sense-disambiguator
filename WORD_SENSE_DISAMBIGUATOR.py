
from time import time
print('\n       [ IMPORTING ALL MODULES ... ]')
start_time = time()

import nltk
import spacy
import textwrap
import numpy as np
import networkx as nx
from colorama import Fore
from nltk.corpus import stopwords
from prettytable import PrettyTable
from nltk.corpus import wordnet as wn
from gensim.models import KeyedVectors

print('          [ ALL MODULES IMPORTED IN {} SECS ]'.format(round(time()-start_time, 3)))
print('\n       [ LOADING PRE-TRAINED GOOGLE\'S WORD-2-VEC MODEL ... ]')
start_time = time()
WORD_2_VEC_MODEL = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=100000)
print('          [ WORD-2-VEC MODEL LOADED IN {} SECS ]'.format(round(time()-start_time, 3)))
NAMED_ENTITY_RECOGNIZER = spacy.load('en_core_web_sm')
TEXT_WRAPPER = textwrap.TextWrapper(width = 50)
TEXT = ""
PR_DAMPING_CONSTANT = 0.85
PR_RANK_PRECISION = 4
SENTENCE_TOKENS = list()
WORD_TOKENS = list()
ALPHABET = "abcdefghijklmnopqrstuvwxyz-"
STOP_WORDS = set(stopwords.words('english'))
COLLOCATION_WINDOW = 4
TEXT_NET = None
SIMILARITY_THRESHOLD = 0.50
SYNONYMOUS_SYNSETS = dict()
TEXT_TO_TOKEN_MAPPING = dict()
NODE_RANK = dict()
DISAMBIGUITY_LOOKUP_TABLE = dict()
TEXT_DOCUMENT_WORD_LIMIT = 2500 #1080
WORDNET_POS_TAG_MAP = {
                        'n': 'Noun',
                        's': 'Adjective',
                        'a': 'Adjective',
                        'r': 'Adverb',
                        'v': 'Verb'   }

def LoadText ( path ) :
    global TEXT
    try : text_file = open(path, 'r')
    except : return False
    TEXT = text_file.read()
    text_file.close()
    return True

def GetWordNetPosTag ( tag ) :
    if tag[0] == 'J' : return 'a'
    return tag[0].lower()

def PreProcessText ( ) :
    global TEXT
    text = TEXT
    named_entities = NAMED_ENTITY_RECOGNIZER(text).ents
    named_entities_names = list()
    for entity in named_entities :
        named_entities_names.append(entity.text.lower())
    
    global WORD_TOKENS, SENTENCE_TOKENS, TEXT_TO_TOKEN_MAPPING
    SENTENCE_TOKENS = list()
    WORD_TOKENS = list()
    TEXT_TO_TOKEN_MAPPING = dict()
    while '--' in text : text = text.replace('--', '-')
    text = text.replace(' -', ' ')
    text = text.replace('- ', ' ')
    
    TEXT = text

    sent_tokens = nltk.tokenize.sent_tokenize(text)
    for sent_token_idx, sent in enumerate(sent_tokens) :
        word_tokens = nltk.tokenize.word_tokenize(sent)
        tagged_words = nltk.pos_tag(word_tokens)
        refined_word_tokens = []
        for word_idx, word_tag in enumerate(tagged_words) :
            word, tag = word_tag
            all_senses = []
            refined_token = ''.join(sym for sym in word.lower() if sym in ALPHABET)
            if ( refined_token in STOP_WORDS or refined_token == '' ) : continue
            if ( refined_token in named_entities_names ) : continue
            if tag[0] in ['N', 'J', 'V', 'R'] and not tag in ['RP', 'NNP', 'NNPS'] :
                synset = wn.synsets(word)
                if len(synset) > 0 :
                    for sense in synset :
                        if sense.pos() == GetWordNetPosTag(tag) :
                            all_senses.append(sense)
            if len(all_senses) > 0 :
                TEXT_TO_TOKEN_MAPPING[(sent_token_idx, word_idx)] = (len(SENTENCE_TOKENS), len(refined_word_tokens))
                refined_word_tokens.append((word, all_senses))
                
        for start_idx in range(len(tagged_words)) :
            for window_length in range(COLLOCATION_WINDOW, 1, -1) :
                if start_idx + window_length - 1 >= len(tagged_words) : continue
                entity_name = ' '.join(word.lower() for word, tag in tagged_words[start_idx:start_idx+window_length])
                if ( entity_name in named_entities_names ) : break
                collocation = '_'.join(word for word, tag in tagged_words[start_idx:start_idx+window_length])
                synset = wn.synsets(collocation)
                if len(synset) > 0 :
                    if (sent_token_idx, start_idx+window_length-1) in TEXT_TO_TOKEN_MAPPING.keys() :
                        TEXT_TO_TOKEN_MAPPING[(sent_token_idx, len(TEXT)+start_idx+window_length-1)] = (len(SENTENCE_TOKENS), len(refined_word_tokens))
                    else :
                        TEXT_TO_TOKEN_MAPPING[(sent_token_idx, start_idx+window_length-1)] = (len(SENTENCE_TOKENS), len(refined_word_tokens))
                    refined_word_tokens.append((collocation.replace('_', ' '), synset))
                    break
        
        if len(refined_word_tokens) > 0 :
            SENTENCE_TOKENS.append(refined_word_tokens)
        WORD_TOKENS += refined_word_tokens

def GetSynsetName ( synset ) :
    return synset.name().split('.')[0]

def ConstructTextNetGraph ( ) :
    global TEXT_NET
    TEXT_NET = nx.Graph()
    AddNodesToTextNet()
    AddEdgesToTextNet()

def AddEdgesToTextNet ( ) :
    CheckSynonymyRelations()
    CheckHypernymyRelations()
    CheckHyponymyRelations()
    CheckMeronymyRelations()
    CheckHolonymyRelations()
    CheckEntailmentRelations()
    CheckCoordinateRelations()
    CheckMiscellaneousRelations()

def AddNodesToTextNet ( ) :
    global TEXT_NET
    for sent_id, sentence in enumerate(SENTENCE_TOKENS) :
        for word_id, word_and_senses in enumerate(sentence) :
            for sense in word_and_senses[1] :
                TEXT_NET.add_node((sent_id, word_id, sense))

def GetSynonymousSynsets ( synset ) :
    global SYNONYMOUS_SYNSETS
    if ( synset.name() in SYNONYMOUS_SYNSETS.keys() ) :
        return SYNONYMOUS_SYNSETS[synset.name()]
    
    SYNONYMOUS_SYNSETS[synset.name()] = set()
    synset_name = GetSynsetName(synset)
    candidate_synonyms_synsets = set()
    syn_words = synset.lemma_names()
    
    if synset_name in syn_words : syn_words.remove(synset_name)
    for word in syn_words :
        candidate_synonyms_synsets = candidate_synonyms_synsets.union(wn.synsets(word))
    
    for syn in candidate_synonyms_synsets :
        if ( GetSynsetName(syn) == synset_name ) : continue
        similarity = syn.wup_similarity(synset)
        if ( similarity is None ) : continue
        if ( similarity >= SIMILARITY_THRESHOLD ) :
            SYNONYMOUS_SYNSETS[synset.name()].add(syn)
    
    return SYNONYMOUS_SYNSETS[synset.name()]

def PageRankAlgorithmOnTextNet ( ) :
    global NODE_RANK
    NODE_RANK = dict()
    all_nodes = list(TEXT_NET.nodes())
    for node in all_nodes :
        NODE_RANK[Hash(node)] = 1 / len(all_nodes)
    
    iteration = 0
    while True :
        iteration += 1
        NODE_RANK_new = dict()
        max_change_in_rank = 0.0
        for node in all_nodes :
            total = 0.0
            neighbours = list(TEXT_NET.neighbors(node))
            for ng in neighbours :
                total += NODE_RANK[Hash(ng)] / len(list(TEXT_NET.neighbors(ng)))
            NODE_RANK_new[Hash(node)] = 1 - PR_DAMPING_CONSTANT + PR_DAMPING_CONSTANT * total
            delta_rank = abs(NODE_RANK_new[Hash(node)]-NODE_RANK[Hash(node)])
            if ( max_change_in_rank < delta_rank ) :
                max_change_in_rank = delta_rank
        NODE_RANK = NODE_RANK_new
        if ( max_change_in_rank < 10**(-1*PR_RANK_PRECISION) ) : break

def Hash ( node ) :
    return (node[0], node[1], node[2].name())

def CheckSynonymyRelations ( ) :
    global TEXT_NET , SYNONYMOUS_SYNSETS
    SYNONYMOUS_SYNSETS = dict()
    SENTENCE_WINDOW_THRESHOLD = 2
    all_synsets = list(TEXT_NET.nodes())
    for node1 in all_synsets :
        for node2 in all_synsets :
            sent1, word1, sense1 = node1
            sent2, word2, sense2 = node2
            if ( sent1 == sent2 and word1 == word2 ) : continue
            if ( abs(sent1-sent2) > SENTENCE_WINDOW_THRESHOLD-1 ) : continue
            if (GetSynsetName(sense1) == GetSynsetName(sense2)) : continue
            if ( AreNodesConnected(node1, node2) ) : continue
            if ( sense1 in GetSynonymousSynsets(sense2) ) :
                TEXT_NET.add_edge(node1, node2)

def CheckHypernymyRelations ( ) :
    global TEXT_NET
    SENTENCE_WINDOW_THRESHOLD = 2
    all_synsets = list(TEXT_NET.nodes())
    for node1 in all_synsets :
        for node2 in all_synsets :
            sent1, word1, sense1 = node1
            sent2, word2, sense2 = node2
            if ( sent1 == sent2 and word1 == word2 ) : continue
            if ( abs(sent1-sent2) > SENTENCE_WINDOW_THRESHOLD-1 ) : continue
            if (GetSynsetName(sense1) == GetSynsetName(sense2)) : continue
            if ( AreNodesConnected(node1, node2) ) : continue
            if ( sense1 in sense2.hypernyms() ) :
                TEXT_NET.add_edge(node1, node2)

def CheckHyponymyRelations ( ) :
    global TEXT_NET
    SENTENCE_WINDOW_THRESHOLD = 2
    all_synsets = list(TEXT_NET.nodes())
    for node1 in all_synsets :
        for node2 in all_synsets :
            sent1, word1, sense1 = node1
            sent2, word2, sense2 = node2
            if ( sent1 == sent2 and word1 == word2 ) : continue
            if ( abs(sent1-sent2) > SENTENCE_WINDOW_THRESHOLD-1 ) : continue
            if (GetSynsetName(sense1) == GetSynsetName(sense2)) : continue
            if ( AreNodesConnected(node1, node2) ) : continue
            if ( sense1 in sense2.hyponyms() ) :
                TEXT_NET.add_edge(node1, node2)

def CheckMeronymyRelations ( ) :
    global TEXT_NET
    SENTENCE_WINDOW_THRESHOLD = 2
    all_synsets = list(TEXT_NET.nodes())
    for node1 in all_synsets :
        for node2 in all_synsets :
            sent1, word1, sense1 = node1
            sent2, word2, sense2 = node2
            if ( sent1 == sent2 and word1 == word2 ) : continue
            if ( abs(sent1-sent2) > SENTENCE_WINDOW_THRESHOLD-1 ) : continue
            if (GetSynsetName(sense1) == GetSynsetName(sense2)) : continue
            if ( AreNodesConnected(node1, node2) ) : continue
            if ( sense1 in sense2.part_meronyms() or sense1 in sense2.substance_meronyms() ) :
                TEXT_NET.add_edge(node1, node2)

def CheckHolonymyRelations ( ) :
    global TEXT_NET
    SENTENCE_WINDOW_THRESHOLD = 2
    all_synsets = list(TEXT_NET.nodes())
    for node1 in all_synsets :
        for node2 in all_synsets :
            sent1, word1, sense1 = node1
            sent2, word2, sense2 = node2
            if ( sent1 == sent2 and word1 == word2 ) : continue
            if ( abs(sent1-sent2) > SENTENCE_WINDOW_THRESHOLD-1 ) : continue
            if (GetSynsetName(sense1) == GetSynsetName(sense2)) : continue
            if ( AreNodesConnected(node1, node2) ) : continue
            if ( sense1 in sense2.part_holonyms() or sense1 in sense2.substance_holonyms() ) :
                TEXT_NET.add_edge(node1, node2)

def CheckEntailmentRelations ( ) :
    global TEXT_NET
    SENTENCE_WINDOW_THRESHOLD = 2
    all_synsets = list(TEXT_NET.nodes())
    for node1 in all_synsets :
        for node2 in all_synsets :
            sent1, word1, sense1 = node1
            sent2, word2, sense2 = node2
            if ( sent1 == sent2 and word1 == word2 ) : continue
            if ( abs(sent1-sent2) > SENTENCE_WINDOW_THRESHOLD-1 ) : continue
            if (GetSynsetName(sense1) == GetSynsetName(sense2)) : continue
            if ( AreNodesConnected(node1, node2) ) : continue
            if ( sense1 in sense2.entailments() ) :
                TEXT_NET.add_edge(node1, node2)

def CheckCoordinateRelations ( ) :
    global TEXT_NET
    SENTENCE_WINDOW_THRESHOLD = 2
    MIN_DEPTH_THRESHOLD = 7
    all_synsets = list(TEXT_NET.nodes())
    for node1 in all_synsets :
        for node2 in all_synsets :
            sent1, word1, sense1 = node1
            sent2, word2, sense2 = node2
            if ( sent1 == sent2 and word1 == word2 ) : continue
            if ( abs(sent1-sent2) > SENTENCE_WINDOW_THRESHOLD-1 ) : continue
            if (GetSynsetName(sense1) == GetSynsetName(sense2)) : continue
            if ( AreNodesConnected(node1, node2) ) : continue
            depths = [ common_hypernym.min_depth() for common_hypernym in sense1.lowest_common_hypernyms(sense2) ]
            for depth in depths :
                if depth >= MIN_DEPTH_THRESHOLD :
                    TEXT_NET.add_edge(node1, node2)
                    break

def CheckMiscellaneousRelations ( ) :
    global TEXT_NET
    SENTENCE_WINDOW_THRESHOLD = 2
    EXCEPTIONAL_SIMILARITY_THRESHOLD = 0.75
    all_synsets = list(TEXT_NET.nodes())
    for node1 in all_synsets :
        for node2 in all_synsets :
            sent1, word1, sense1 = node1
            sent2, word2, sense2 = node2
            if ( sent1 == sent2 and word1 == word2 ) : continue
            if ( abs(sent1-sent2) > SENTENCE_WINDOW_THRESHOLD-1 ) : continue
            if (GetSynsetName(sense1) == GetSynsetName(sense2)) : continue
            if ( AreNodesConnected(node1, node2) ) : continue
            similarity = sense1.wup_similarity(sense2)
            if ( not similarity is None and similarity >= EXCEPTIONAL_SIMILARITY_THRESHOLD ) :
                TEXT_NET.add_edge(node1, node2)

def AreNodesConnected ( node1 , node2 ) :
    return node1 in TEXT_NET.neighbors(node2)

def IsAlphaNumeric ( word ) :
    word = word.lower()
    for letter in ALPHABET + "0123456789" :
        if letter in word :
            return True
    return False

def GetHighlightColour ( sentence_id , word_id ) :
    if ( len(SENTENCE_TOKENS[sentence_id][word_id][1]) == 1 ) :
        return Fore.GREEN
    return Fore.MAGENTA

def DisplayAnnotatedText ( ) :
    sentence_tokens = nltk.tokenize.sent_tokenize(TEXT)
    annotated_text = ""
    counter = 0
    global DISAMBIGUITY_LOOKUP_TABLE
    DISAMBIGUITY_LOOKUP_TABLE = dict()
    double_quotation_inside = False
    single_quotation_inside = False
    last_symbol = ''
    current_line_length = 0
    print('\n\t', end='')
    for sent_token_idx, sentence in enumerate(sentence_tokens) :
        word_tokens = nltk.tokenize.word_tokenize(sentence)
        for word_token_idx, word in enumerate(word_tokens) :
            old_ann_text_length = len(annotated_text)
            word = word.replace('``', '"').replace("''", '"')
            flag = False
            custom_space = ' '
            if ( current_line_length == 0 ) : custom_space = ''
            elif last_symbol == '"' and double_quotation_inside : custom_space = ''
            elif last_symbol == "'" and single_quotation_inside : custom_space = ''
            elif last_symbol in ['[', '(', '{'] : custom_space = ''
            if (sent_token_idx, word_token_idx) in TEXT_TO_TOKEN_MAPPING.keys() :
                counter += 1
                mapped_sent, mapped_word = TEXT_TO_TOKEN_MAPPING[(sent_token_idx, word_token_idx)]
                DISAMBIGUITY_LOOKUP_TABLE[counter] = (mapped_sent, mapped_word)
                annotated_text += custom_space + word + '[*' + str(counter) + '*]'
                print( custom_space + word + GetHighlightColour(mapped_sent, mapped_word) + '[*' + str(counter) + '*]' + Fore.WHITE , end = '' )
                last_symbol = ']'
                flag = True
            
            if (sent_token_idx, len(TEXT)+word_token_idx) in TEXT_TO_TOKEN_MAPPING.keys() :
                counter += 1
                mapped_sent, mapped_word = TEXT_TO_TOKEN_MAPPING[(sent_token_idx, len(TEXT)+word_token_idx)]
                DISAMBIGUITY_LOOKUP_TABLE[counter] = (mapped_sent, mapped_word)
                if not flag : 
                    annotated_text += custom_space + word
                    print( custom_space + word , end = '' )
                    last_symbol = word[-1]
                annotated_text += '[*' + str(counter) + '*]'
                print( GetHighlightColour(mapped_sent, mapped_word) + '[*' + str(counter) + '*]' + Fore.WHITE , end = '' )
                last_symbol = ']'
                flag = True
            
            if not flag :
                if IsAlphaNumeric(word) : 
                    annotated_text += ' ' + word
                    print( custom_space + word , end = '' )
                    last_symbol = word[-1]
                elif word[0] == '"' :
                    if double_quotation_inside :
                        print( word , end = '' )
                        double_quotation_inside = False
                    else :
                        print( custom_space + word , end = '' )
                        double_quotation_inside = True
                elif word[0] == "'" :
                    if single_quotation_inside :
                        print( word , end = '' )
                        single_quotation_inside = False
                    else :
                        print( custom_space + word , end = '' )
                        single_quotation_inside = True
                elif word[0] == "[" : print( custom_space + word , end = '' )
                elif word[0] == "(" : print( custom_space + word , end = '' )
                elif word[0] == "{" : print( custom_space + word , end = '' )
                elif word[0] == "]" : print( word , end = '' )
                elif word[0] == ")" : print( word , end = '' )
                elif word[0] == "}" : print( word , end = '' )
                else : 
                    annotated_text += word
                    print( word , end = '' )
                last_symbol = word[-1]
            current_line_length += len(annotated_text) - old_ann_text_length
            if ( current_line_length >= 110 ) :
                current_line_length = 0
                print('\n\t', end='')

def GetTokenEmbedding ( word ) :
    phrase = word.replace('_', ' ')
    tokens = nltk.tokenize.word_tokenize(phrase)
    valid_tokens = 0
    net_embedding = np.array([0.0]*300, dtype='float32')
    for token in tokens :
        try :
            embedding = WORD_2_VEC_MODEL[token]
            valid_tokens += 1
            net_embedding += embedding
        except :
            try :
                embedding = WORD_2_VEC_MODEL[token.upper()]
                valid_tokens += 1
                net_embedding += embedding
            except :
                try :
                    embedding = WORD_2_VEC_MODEL[token[0].upper()+token[1:]]
                    valid_tokens += 1
                    net_embedding += embedding
                except :
                    continue
    if ( valid_tokens == 0 ) : return np.array([10**10]*300)
    return net_embedding / valid_tokens

def GetRefinedTokens ( sentence ) :
    words = nltk.tokenize.word_tokenize(sentence)
    refined_tokens = list()
    for word in words :
        refined_token = ''.join(sym for sym in word if sym.lower() in ALPHABET)
        if ( refined_token.lower() in STOP_WORDS or refined_token == '' ) : continue
        refined_tokens.append(refined_token)
    return refined_tokens

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

def SortDictionary ( dictionary ) :
    return dict(sorted(dictionary.items(), key=lambda item: (item[0][0], item[0][1], -1*item[1])))

def MostCertainMeaning ( sent_loc , word_loc ) :
    SUB_DICT = [(k[2], v) for k, v in NODE_RANK.items() if k[0]==sent_loc and k[1]==word_loc]
    maximum = SUB_DICT[0][1]
    top_sense = wn.synset(SUB_DICT[0][0])
    for sense_name, rank in SUB_DICT[1:] :
        if ( rank < maximum ) : break
        top_sense = BreakTies((sent_loc, word_loc, SENTENCE_TOKENS[sent_loc][word_loc][0]), wn.synset(sense_name), top_sense)
        maximum = rank
    return top_sense

def GenerateResultTable ( ) :
    RESULTS = PrettyTable(["AMBIGUITY ID", "WORD / COLLOCATION", "DEDUCED PART OF SPEECH", "DEDUCED DEFINITION"])
    RESULTS.align["DEDUCED DEFINITION"] = "l"
    for disambig_id in range(1, len(DISAMBIGUITY_LOOKUP_TABLE)+1) :    
        sent_loc, word_loc = DISAMBIGUITY_LOOKUP_TABLE[disambig_id]
        most_suitable_sense = MostCertainMeaning(sent_loc, word_loc)
        definition = most_suitable_sense.definition()
        definition = TEXT_WRAPPER.fill(text=definition)
        part_of_speech = WORDNET_POS_TAG_MAP[most_suitable_sense.pos()]
        word_or_coll = SENTENCE_TOKENS[sent_loc][word_loc][0]
        disambig_id = ' ' * (3-len(str(disambig_id))) + str(disambig_id)
        RESULTS.add_row([disambig_id, word_or_coll, part_of_speech, definition])
    return RESULTS

print('\n\n\n\t\t +++ WORD SENSE DISAMBIGUATOR +++')
while True :
    file_path = input('\n\n    ENTER TEXT DOCUMENT PATH : ')
    if ( not LoadText(file_path) ) :
        print( "\n       [ TEXT DOCUMENT COULD NOT BE READ SUCCESFULLY ! ] " )
        continue
    word_size = len(nltk.tokenize.word_tokenize(TEXT))
    if ( word_size > TEXT_DOCUMENT_WORD_LIMIT ) :
        print( "\n       [ TEXT DOCUMENT EXCEEDS THE WORD LIMIT ! ] " )
        continue
    
    print('\n       [ PROCESSING TEXT DOCUMENT ... ]')
    start_time = time()
    PreProcessText()
    print('          [ DOCUMENT PROCESSED IN {} SECS ]'.format(round(time()-start_time, 3)))
    
    print('\n       [ CONSTRUCTING DISAMBIGUATION NETWORK ... ]')
    start_time = time()
    ConstructTextNetGraph()
    print('          [ NETWORK CONSTRUCTED IN {} SECS ]'.format(round(time()-start_time, 3)))
    
    print('\n       [ PERFORMING PAGE-RANK-STYLE ALGORITHM ... ]')
    start_time = time()
    PageRankAlgorithmOnTextNet()
    print('          [ ALGORITHM FINISHED IN {} SECS ]'.format(round(time()-start_time, 3)))

    print('\n         < AMBIGUOUS WORDS/COLLOCATIONS ARE ANNOTATED WITH A COLOURED ID AND MARKER >') 
    print('         < MAGENTA COLOUR INDICATES WORDS/COLLOCATIONS WITH MULTIPLE CANDIDATE DEFINITIONS >') 
    print('         < GREEN COLOUR INDICATES WORDS/COLLOCATIONS WITH UNIQUE CANDIDATE DEFINITION >') 
    DisplayAnnotatedText()

    print('\n\n       [ GENERATING FINAL RESULTS ... ]')
    start_time = time()
    result_table = GenerateResultTable()
    print('          [ RESULTS GENERATED IN {} SECS ]'.format(round(time()-start_time, 3)))
    print('\n', end='')
    print(result_table)
