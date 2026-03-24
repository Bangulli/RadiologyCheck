from lingua import Language, LanguageDetectorBuilder

class LangDct():
    def __init__(self, enable=True):
        self.languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH]
        self.target_language = Language.ENGLISH
        self.detector = LanguageDetectorBuilder.from_languages(*self.languages).build()
        self.enable = enable
        
    def needs_translation(self, text):
        if self.enable: return not (self.detector.detect_language_of(text) == self.target_language)
        else: return False
    
    def __call__(self, text):
        return self.needs_translation(text)