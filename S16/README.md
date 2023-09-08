# The School of AI - ERA(Extensive & Reimagined AI Program) - Assignment 16

This folder consists of Assignment-16 from ERA course offered by - TSAI(The school of AI). 
Follow https://theschoolof.ai/ for more updates on TSAI

Assignment-16
We now move to transformer based learning from CNN's. This assignment is an extenstion of Assignment-15. While Assignment 15 consisted of a naive transformer model for Italian - English translation, Assignment 16 consists of a sped-up version of a similar transformer model, this time trained for French - English translation.

Here are some requirements defined by the assignment:
- Pick the "en-fr" dataset from opus_books
- Remove all English sentences with more than 150 "tokens"
- Remove all french sentences where len(fench_sentences) > len(english_sentrnce) + 10
- Train your own transformer (E-D) (do anything you want, use PyTorch, OCP, PS, AMP, etc), but get your loss under 1.8
- Last, but not the least - Enjoy!!


### RESULTS:

Train Accuracy:
Mean training loss at end of epoch 41 = 1.6602355853539892

Translation during validation:
--------------------------------------------------------------------------------
    SOURCE: "Conseil!" I shouted. "Conseil!"
    TARGET: « Conseil, m'écriai-je, Conseil !
    PREDICTED: « Conseil ! » criai - je .
--------------------------------------------------------------------------------
    SOURCE: My father stopped, looked at me disdainfully, and contented himself with saying, "I believe you are mad."
    TARGET: Mon père s'arrêta, me regarda avec dédain, et se contenta de me répondre: --Vous êtes fou, je crois.
    PREDICTED: Mon père s ' arrêta , me regarda d ' un oeil dédaigneux , et se contenta de dire :
--------------------------------------------------------------------------------


