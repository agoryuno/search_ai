
## Description

This is a simple example of using OpenAI Embeddings API to train a specialized model. The repository contains everything you need 
to train a specialized model that extracts date information from free-form search queries. The model takes as input embedding vectors produced
by calling the [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings) with the text of the query and outputs a date to start a
search from, extracted from the free-form instruction in the query and the current date included both in the embedding and prepended to the embedding
vector as three floating point numbers: the year (decade and year, i.e. the two last digits of the year), month, and day scaled by dividing by 100.

For example, with a query like:

  "Give me a summary of the last week. Just everything you find."
  
if the current date is 2023-03-20, the model should (ideally) return "2023-03-13" - the current date minus a week (7 days).

## Motivation

I wanted to see if and how OpenAI Embeddings could be used to distill a specialized model. Overall, the Embeddings API model is perfectly usable
for that purpose with some caveats.

In particular, the initital hypothesis that inserting the current date as a 
 simple date stamp into the embedded query, e.g. "<2023-03-20>", would be sufficient to make the embedding model "figure it out", was
proven incorrect. However, "spelling" out a full date with the week day included, e.g. "Today is Wed, March 20, 2023." was enough to tip the embedder
off and get the required information into the embedding vector so it could be extracted by the distilled model.

## Training and use

All code for generating training and validation data and training the model is in the Jupyter Notebooks inside the search_ai folder.
A pickled list with the full dataset used for training is also included so the model can be trained without needing an OpenAI account.
While the model is small enough to fit into the memory of even the most modest of laptops, training on the CPU is not advised as it is 
catastrophically slow.

In Google Colab the model takes no more than 30 minutes to train.

Inference can be done on either the GPU or the CPU.

## Results

The trained model gives 99% accuracy on the training and validation datasets. However, the two heavily intersect due to the nature of the process
used to generate the training data. More training prompts are needed before assessing the true practical fitness of the model.


## WIP notes on CallFormer

1. All arguments to commands must be enclosed in double quotes, including numeric arguments. This is to simplify processing and allow the model to terminate an argument
at any time: quotes delimit the start and end of an argument's value and allow the model to decide when it has reached the end of the argument's value. Parsing of values is left up to the recepient of the command.

2. All arguments to commands are always required - optional arguments are not supported. If you need a command to take a variable number of arguments, make several versions of the command with different numbers of arguments and different names. For instance a "search" command can take the start and end dates, as well as the maximum number of results to return, as arguments. This could then have versions: search(start, end, max_results), search1(start), search2(start, end), search3(max_results) and so on.

Of course an alternative (and possibly - better) solution for that example would be to have a dedicated "limit" command that takes the maximum number of results to return as an argument.

3. Floating point numbers in arguments of commands that are between -1 and 1 should have a
zero before the decimal point, e.g. 0.5, 0.25, -0.75, 0.125, 0.0625, etc. 