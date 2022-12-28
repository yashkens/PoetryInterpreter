# What did the author mean?
Language model generates poems and then tries to explain their meaning.
My project for [NaNoGenMO](https://github.com/NaNoGenMo/2022) :)

**[Usage Example](https://github.com/yashkens/PoetryInterpreter/blob/main/try_generation.ipynb)**  
The notebook shows how to run poem or analysis generation separately and how to use them in a pipeline.

* Poem Generation Models were tuned on imagist and modern poems scraped from [Poem Hunter](https://www.poemhunter.com/).
* Model for poem analysis from was tuned on data from [Literary Devices](https://literarydevices.net/poem-analysis/).
* GPT-J finishes the analysis, as the tuned model struggles to make analysis long enough.
