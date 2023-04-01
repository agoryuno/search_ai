import unittest
import sys
sys.path.append('..')

import numpy as np

from callformer.commands import Date, SearchNotesCommand1
from callformer.tokenizer import Tokenizer


class TestSearchNotesCommand1(unittest.TestCase):
    
        def test_1(self):
            comm = SearchNotesCommand1()
            comm2 = SearchNotesCommand1()
            self.assertEqual(id(comm.tokenizer), id(comm2.tokenizer))
            self.assertNotEqual(id(comm.args_list), id(comm2.args_list))

        def test_2(self):
            comm = SearchNotesCommand1()
            for toks in comm.valid_tokens():
                idx = np.random.randint(0, len(toks))
                comm.add_token(toks[idx])
            print ("".join([comm.tokenizer.vocab[t] for t in comm.sequence]))

class TestDateArgument(unittest.TestCase):

    def test_1(self):
        date = Date()
        for toks in date.valid_tokens():
            idx = np.random.randint(0, len(toks))
            date.add_token(toks[idx])
            #print ([date.tokenizer.vocab[t] for t in toks],
            #        " -> ", 
            #        date.tokenizer.vocab[toks[idx]])


class TestArgumentList(unittest.TestCase):
         
        def test_1(self):
            comm = SearchNotesCommand1()
            alist = comm.args_list
            for toks in alist.valid_tokens():
                idx = np.random.randint(0, len(toks))
                alist.add_token(toks[idx])
                #print ([alist.tokenizer.vocab[t] for t in toks],
                #        " -> ", 
                #        alist.tokenizer.vocab[toks[idx]])
            print ("".join([alist.tokenizer.vocab[t] for t in alist.sequence]))

# run the tests
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
