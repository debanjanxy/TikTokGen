import yaml
from text_generator.text_gen import TextGenerator
# from audio_generator.aud_gen import AudioGenerator



with open('config/config.yml', 'r') as file:
    config = yaml.safe_load(file)
tgen = TextGenerator(config)
#TODO: Write a query constructor using langchain
query = "Share 5 motivational quote. End it with a newline character. Don't generate any introductory line."
results = tgen.query(query)
# agen = AudioGenerator()
for i, result in enumerate(results):
    print(result)
    # filename = "audio_{i}.wav"
    # agen.tts(result, filename)




