#moses code normalization......................................

import re 
import unicodedata 

class MosesNormalizer:
  QUOTES = {'“': '"', '”': '"', '‘': "'", '’': "'"}
  DASHES = {'–': '-', '—': '-'}

  def normalize(self, text:str):

    for orig, repl in self.QUOTES.items():
      text = text.replace(orig, repl)

    for orig,repl in self.DASHES.items():
         text = text.replace(orig, repl)

    text = re.sub(r'[\u200b\ufeff]', '', text)   

    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', text)  
    
    text = unicodedata.normalize('NFC', text)
      
    return text.strip()


normalizer = MosesNormalizer()
print(normalizer.normalize("Hello , world ! “Nice”—to meet you…"))

#indic nlp...............................................................

import unicodedata
import re 

class IndicNormalizer:
    
      NUKTA_pattern = re.compile(r'(\w)\u093c')

      def normalize(self, text:str):
           
          text = unicodedata.normalize('NFC', text)
          

          text = self.NUKTA_pattern.sub(lambda m: unicodedata.normalize('NFC', m.group(0)), text )

          text = re.sub(r'[\u200c\u200d]+', '', text)

          return text.strip()

txt = 'hello world'

print(txt.strip("--"))
