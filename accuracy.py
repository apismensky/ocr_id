import Levenshtein


#  Levenshtein distance
def calculate_char_accuracy(detected_text, truth_text):
    return 100 - Levenshtein.distance(detected_text.lower(), truth_text.lower()) * 100 / len(truth_text)


def calculate_word_accuracy(detected_text, truth_text):
    detected_words = detected_text.lower().split()
    truth_words = truth_text.lower().split()
    total_words = len(truth_words)
    correct_words = sum(1 for dw in detected_words if dw in truth_words)
    return (correct_words / total_words) * 100


def main():
    detected_text_tesseract = "4d DL 999 as = Ne allo) 2NICK © , q 12 RESTR oe } lick: 5 DD 8888888888 1234 SZ"
    detected_text_easyocr = '''9 , ARKANSAS DRIVER'S LICENSE CLAss D 4d DLN 999999999 3 DOB 03/05/1960 ] 2 SCKPLE 123 NORTH STREET CITY AR 12345 ISS 4b EXP 03/05/2018 03/05/2026 15 SEX 16 HGT 18 EYES 5'-10" BRO 9a END NONE 12 RESTR NONE Ylck Sorble DD 8888888888 1234 THE'''
    truth_text = """USA ARKANSAS 
DRIVER'S LICENSE
9 CLASS D
4d DLN 999999999
3 DOB 03/05/1960
1 SAMPLE 
2 NICK
GREAT SEAL OF THE STATE OF ARKANSAS
8 123 NORTH STREET
CITY, AR 12345
4a ISS
03/05/2018
4b EXP 
03/05/2026
NS60
15 SEX 16 HGT 18 EYES
M 5'-10'' BRO
9a END NONE
12 RESTR NONE
DD 8888888888 1234
Nick Sample"""

    word_accuracy_tesseract = calculate_word_accuracy(detected_text_tesseract, truth_text)
    char_accuracy_tesseract = calculate_char_accuracy(detected_text_tesseract, truth_text)
    word_accuracy_easyocr = calculate_word_accuracy(detected_text_easyocr, truth_text)
    char_accuracy_easyocr = calculate_char_accuracy(detected_text_easyocr, truth_text)
    print(f"Word-level accuracy (case-insensitive) tesseract: {word_accuracy_tesseract:.2f}%, easyocr: {word_accuracy_easyocr:.2f}%")
    print(f"Character-level accuracy (case-insensitive) tesseract: {char_accuracy_tesseract:.2f}%, easyocr: {char_accuracy_easyocr:.2f}%")


if __name__ == "__main__":
    main()