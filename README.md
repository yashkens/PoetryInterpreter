# What did the author mean?
Language model generates poems and then tries to explain their meaning.

**Note:** I will create a pipeline later. For now, here is the working code done in a hurry to complete the project in 
time for [NaNoGenMO](https://github.com/NaNoGenMo/2022) :)

1. **Generate Poem**  
Model tuned on imagist and modern poems scraped from [Poem Hunter](https://www.poemhunter.com/).
2. **Start Analysis**  
Model tuned on poem analyzes from [Literary Devices](https://literarydevices.net/poem-analysis/).
It's good to begin an analysis, but struggles to make it long enough.
3. **Finish Analysis**  
GPT-J with prompt from the previous two steps. Simple heuristics make it sound a little more
irritated and force the analysis to end logically.
4. **Format Output for Latex**  
Make poem texts italic. Find key phrases with Rake algorithm and make them bold.
Add new lines and new pages.