"""
Comprehensive Multilingual Example Database for Cross-Lingual NER.

Provides extensive sample sentences categorized by target language and dataset source
(IndicNER / WikiANN, CoNLL-2003 Standard, Synthetic Agglutinative, Cultural & Historical).
"""

from typing import List, Dict, Any

# Dataset category names
DATASET_CATEGORIES = [
    "All Datasets",
    "IndicNER / WikiANN (Real)",
    "CoNLL-2003 Standard",
    "Synthetic Agglutinative Corpus",
    "Cultural & Historical Texts"
]

LANGUAGE_LIST = [
    "Bhojpuri",
    "Maithili",
    "Santali",
    "Hindi",
    "Marathi",
    "Bengali",
    "Gujarati",
    "Odia",
    "Punjabi",
    "Other / English"
]

# Multilingual example database
MULTILINGUAL_EXAMPLES: Dict[str, List[Dict[str, Any]]] = {
    "Bhojpuri": [
        {
            "id": "bhoj_1",
            "text": "डॉक्टर राजेंद्र प्रसाद का जन्म जीरादेई में भइल रहे।",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Historical figure and location in Bhojpuri",
            "tokens": ["डॉक्टर", "राजेंद्र", "प्रसाद", "का", "जन्म", "जीरादेई", "में", "भइल", "रहे।"],
            "tags": ["O", "B-PER", "I-PER", "O", "O", "B-LOC", "O", "O", "O"]
        },
        {
            "id": "bhoj_2",
            "text": "पटना विश्वविद्यालय बिहार के सभसे पुराना संस्थान ह।",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Educational institution & state location",
            "tokens": ["पटना", "विश्वविद्यालय", "बिहार", "के", "सभसे", "पुराना", "संस्थान", "ह।"],
            "tags": ["B-ORG", "I-ORG", "B-LOC", "O", "O", "O", "O", "O"]
        },
        {
            "id": "bhoj_3",
            "text": "भोजपुरी साहित्य अकादमी पटनामें स्थित बा।",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Agglutinative postposition 'पटनामें'",
            "tokens": ["भोजपुरी", "साहित्य", "अकादमी", "पटनामें", "स्थित", "बा।"],
            "tags": ["B-ORG", "I-ORG", "I-ORG", "B-LOC", "O", "O"]
        },
        {
            "id": "bhoj_4",
            "text": "वीर कुँवर सिंह 1857 के क्रांति में जगदीशपुर से नेतृत्व कइले रहलें।",
            "dataset": "CoNLL-2003 Standard",
            "description": "Freedom fighter with temporal expression & location",
            "tokens": ["वीर", "कुँवर", "सिंह", "1857", "के", "क्रांति", "में", "जगदीशपुर", "से", "नेतृत्व", "कइले", "रहलें।"],
            "tags": ["B-PER", "I-PER", "I-PER", "B-DATE", "O", "O", "O", "B-LOC", "O", "O", "O", "O"]
        },
        {
            "id": "bhoj_5",
            "text": "महेंद्र मिसिर के सिवान में बहुते मान-सम्मान मिलल।",
            "dataset": "CoNLL-2003 Standard",
            "description": "Folk poet & location in Bhojpuri",
            "tokens": ["महेंद्र", "मिसिर", "के", "सिवान", "में", "बहुते", "मान-सम्मान", "मिलल।"],
            "tags": ["B-PER", "I-PER", "O", "B-LOC", "O", "O", "O", "O"]
        },
        {
            "id": "bhoj_6",
            "text": "हम बिरसा मुंडा से पटनामें भेंट भईल रहलीं।",
            "dataset": "Synthetic Agglutinative Corpus",
            "description": "Person entity with postposition attached to location",
            "tokens": ["हम", "बिरसा", "मुंडा", "से", "पटनामें", "भेंट", "भईल", "रहलीं।"],
            "tags": ["O", "B-PER", "I-PER", "O", "B-LOC", "O", "O", "O"]
        },
        {
            "id": "bhoj_7",
            "text": "सर्चलाइट समाचार पत्र के कार्यालय बक्सर में खुलल रहल।",
            "dataset": "Synthetic Agglutinative Corpus",
            "description": "Media organization & district location",
            "tokens": ["सर्चलाइट", "समाचार", "पत्र", "के", "कार्यालय", "बक्सर", "में", "खुलल", "रहल।"],
            "tags": ["B-ORG", "I-ORG", "I-ORG", "O", "O", "B-LOC", "O", "O", "O"]
        },
        {
            "id": "bhoj_8",
            "text": "भीखारी ठाकुर के जन्म सारण जिला के कुतुबपुर गाँव में भइल रहल।",
            "dataset": "Cultural & Historical Texts",
            "description": "Bhojpuri playwright & village location",
            "tokens": ["भीखारी", "ठाकुर", "के", "जन्म", "सारण", "जिला", "के", "कुतुबपुर", "गाँव", "में", "भइल", "रहल।"],
            "tags": ["B-PER", "I-PER", "O", "O", "B-LOC", "O", "O", "B-LOC", "O", "O", "O", "O"]
        }
    ],

    "Maithili": [
        {
            "id": "mai_1",
            "text": "महाकवि विद्यापति मिथिलाक महान कवि छलाह।",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Classical Maithili poet and region",
            "tokens": ["महाकवि", "विद्यापति", "मिथिलाक", "महान", "कवि", "छलाह।"],
            "tags": ["O", "B-PER", "B-LOC", "O", "O", "O"]
        },
        {
            "id": "mai_2",
            "text": "दरभंगा राज किला बिहार राज्यक दरभंगा जिलामें अछि।",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Historical fort and district location with postposition",
            "tokens": ["दरभंगा", "राज", "किला", "बिहार", "राज्यक", "दरभंगा", "जिलामें", "अछि।"],
            "tags": ["B-LOC", "I-LOC", "I-LOC", "B-LOC", "O", "B-LOC", "O", "O"]
        },
        {
            "id": "mai_3",
            "text": "मैथिली साहित्य परिषद मधुबनीमे नया भवन बनौने अछि।",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Literary organization in Madhubani",
            "tokens": ["मैथिली", "साहित्य", "परिषद", "मधुबनीमे", "नया", "भवन", "बनौने", "अछि।"],
            "tags": ["B-ORG", "I-ORG", "I-ORG", "B-LOC", "O", "O", "O", "O"]
        },
        {
            "id": "mai_4",
            "text": "जनकपुर धाम नेपाल देशक प्रसिद्ध धार्मिक स्थल अछि।",
            "dataset": "CoNLL-2003 Standard",
            "description": "Cross-border location (Janakpur, Nepal)",
            "tokens": ["जनकपुर", "धाम", "नेपाल", "देशक", "प्रसिद्ध", "धार्मिक", "स्थल", "अछि।"],
            "tags": ["B-LOC", "I-LOC", "B-LOC", "O", "O", "O", "O", "O"]
        },
        {
            "id": "mai_5",
            "text": "विद्यापति मैथिली भाषा के महान कवि छलाह और ओ दरभंगामे रहैत छलाह।",
            "dataset": "Synthetic Agglutinative Corpus",
            "description": "Person, language culture tag & postposition location",
            "tokens": ["विद्यापति", "मैथिली", "भाषा", "के", "महान", "कवि", "छलाह", "और", "ओ", "दरभंगामे", "रहैत", "छलाह।"],
            "tags": ["B-PER", "B-MISC", "I-MISC", "O", "O", "O", "O", "O", "O", "B-LOC", "O", "O"]
        },
        {
            "id": "mai_6",
            "text": "सामा चकेवा पर्व कार्तिक पूर्णिमा को मधुबनीमे मनाओल जाइछ।",
            "dataset": "Cultural & Historical Texts",
            "description": "Cultural festival, date expression & location",
            "tokens": ["सामा", "चकेवा", "पर्व", "कार्तिक", "पूर्णिमा", "को", "मधुबनीमे", "मनाओल", "जाइछ।"],
            "tags": ["B-MISC", "I-MISC", "I-MISC", "B-DATE", "I-DATE", "O", "B-LOC", "O", "O"]
        },
        {
            "id": "mai_7",
            "text": "फणीश्वर नाथ रेणु पूर्णिया जिलाक हिंगना गाँवमे जनम लेने छलाह।",
            "dataset": "Cultural & Historical Texts",
            "description": "Renowned writer & birthplace location",
            "tokens": ["फणीश्वर", "नाथ", "रेणु", "पूर्णिया", "जिलाक", "हिंगना", "गाँवमे", "जनम", "लेने", "छलाह।"],
            "tags": ["B-PER", "I-PER", "I-PER", "B-LOC", "O", "B-LOC", "O", "O", "O", "O"]
        }
    ],

    "Santali": [
        {
            "id": "san_1",
            "text": "ᱫᱤᱥᱚᱢ ᱜᱩᱨᱩ ᱥᱤᱵᱩ ᱥᱚᱨᱮᱱ ᱨᱟanchi ᱨᱮ ᱢᱮᱱᱟᱭᱟ।",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Ol Chiki script: leader & Ranchi location",
            "tokens": ["ᱫᱤᱥᱚᱢ", "ᱜᱩᱨᱩ", "ᱥᱤᱵᱩ", "ᱥᱚᱨᱮᱱ", "ᱨᱟanchi", "ᱨᱮ", "ᱢᱮᱱᱟᱭᱟ।"],
            "tags": ["O", "O", "B-PER", "I-PER", "B-LOC", "O", "O"]
        },
        {
            "id": "san_2",
            "text": "Sidhu Murmu ada Santhal Pargana re janam holena.",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Latin-script Santali: Freedom fighter & region",
            "tokens": ["Sidhu", "Murmu", "ada", "Santhal", "Pargana", "re", "janam", "holena."],
            "tags": ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "O", "O", "O"]
        },
        {
            "id": "san_3",
            "text": "Sido Kanhu Murmu University Dumka re menak-a.",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "University organization & Dumka town",
            "tokens": ["Sido", "Kanhu", "Murmu", "University", "Dumka", "re", "menak-a."],
            "tags": ["B-ORG", "I-ORG", "I-ORG", "I-ORG", "B-LOC", "O", "O"]
        },
        {
            "id": "san_4",
            "text": "Birsa Munda Ranchi re Ulgulan aythenaye.",
            "dataset": "CoNLL-2003 Standard",
            "description": "Tribal leader Birsa Munda in Ranchi",
            "tokens": ["Birsa", "Munda", "Ranchi", "re", "Ulgulan", "aythenaye."],
            "tags": ["B-PER", "I-PER", "B-LOC", "O", "B-MISC", "O"]
        },
        {
            "id": "san_5",
            "text": "सिद्धू कान्हू दुमकाते सेनावकेना और सोहराय पर्व मनावया।",
            "dataset": "Synthetic Agglutinative Corpus",
            "description": "Devanagari Santali: Leaders, agglutinative 'दुमकाते' & festival",
            "tokens": ["सिद्धू", "कान्हू", "दुमकाते", "सेनावकेना", "और", "सोहराय", "पर्व", "मनावया।"],
            "tags": ["B-PER", "I-PER", "B-LOC", "O", "O", "B-MISC", "I-MISC", "O"]
        },
        {
            "id": "san_6",
            "text": "Sohrai Parab Santal Pargana re raska tebon manao-a.",
            "dataset": "Cultural & Historical Texts",
            "description": "Traditional Santal festival in Santhal Pargana",
            "tokens": ["Sohrai", "Parab", "Santal", "Pargana", "re", "raska", "tebon", "manao-a."],
            "tags": ["B-MISC", "I-MISC", "B-LOC", "I-LOC", "O", "O", "O", "O"]
        },
        {
            "id": "san_7",
            "text": "Draupadi Murmu Mayurbhanj re janam lenaye.",
            "dataset": "Cultural & Historical Texts",
            "description": "President of India & Mayurbhanj district",
            "tokens": ["Draupadi", "Murmu", "Mayurbhanj", "re", "janam", "lenaye."],
            "tags": ["B-PER", "I-PER", "B-LOC", "O", "O", "O"]
        }
    ],

    "Hindi": [
        {
            "id": "hin_1",
            "text": "डॉ. एपीजे अब्दुल कलाम का जन्म रामेश्वरम में हुआ था।",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Former President of India & Rameswaram location",
            "tokens": ["डॉ.", "एपीजे", "अब्दुल", "कलाम", "का", "जन्म", "रामेश्वरम", "में", "हुआ", "था।"],
            "tags": ["O", "B-PER", "I-PER", "I-PER", "O", "O", "B-LOC", "O", "O", "O"]
        },
        {
            "id": "hin_2",
            "text": "भारतीय अंतरिक्ष अनुसंधान संगठन (ISRO) का मुख्यालय बेंगलुरु में स्थित है।",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Space agency organization & Bengaluru headquarters",
            "tokens": ["भारतीय", "अंतरिक्ष", "अनुसंधान", "संगठन", "(ISRO)", "का", "मुख्यालय", "बेंगलुरु", "में", "स्थित", "है।"],
            "tags": ["B-ORG", "I-ORG", "I-ORG", "I-ORG", "B-ORG", "O", "O", "B-LOC", "O", "O", "O"]
        },
        {
            "id": "hin_3",
            "text": "आईआईटी दिल्ली ने 15 अगस्त को नया एआई केंद्र शुरू किया।",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Academic institute, date tag & technology center",
            "tokens": ["आईआईटी", "दिल्ली", "ने", "15", "अगस्त", "को", "नया", "एआई", "केंद्र", "शुरू", "किया।"],
            "tags": ["B-ORG", "I-ORG", "O", "B-DATE", "I-DATE", "O", "O", "O", "O", "O", "O"]
        },
        {
            "id": "hin_4",
            "text": "बिरसा मुंडा ने 1900 में रांची के खूंटी इलाके में उलगुलान आंदोलन शुरू किया था।",
            "dataset": "CoNLL-2003 Standard",
            "description": "Freedom struggle figure, date & location",
            "tokens": ["बिरसा", "मुंडा", "ने", "1900", "में", "रांची", "के", "खूंटी", "इलाके", "में", "उलगुलान", "आंदोलन", "शुरू", "किया", "था।"],
            "tags": ["B-PER", "I-PER", "O", "B-DATE", "O", "B-LOC", "O", "B-LOC", "O", "O", "B-MISC", "I-MISC", "O", "O", "O"]
        },
        {
            "id": "hin_5",
            "text": "टाटा स्टील कंपनी की स्थापना जमशेदपुर में हुई थी।",
            "dataset": "CoNLL-2003 Standard",
            "description": "Industrial company & city location",
            "tokens": ["टाटा", "स्टील", "कंपनी", "की", "स्थापना", "जमशेदपुर", "में", "हुई", "थी।"],
            "tags": ["B-ORG", "I-ORG", "I-ORG", "O", "O", "B-LOC", "O", "O", "O"]
        },
        {
            "id": "hin_6",
            "text": "बिरसा मुंडा ने रांचीमें महान आंदोलन की शुरुआत की थी।",
            "dataset": "Synthetic Agglutinative Corpus",
            "description": "Agglutinative 'रांचीमें' postposition test",
            "tokens": ["बिरसा", "मुंडा", "ने", "रांचीमें", "महान", "आंदोलन", "की", "शुरुआत", "की", "थी।"],
            "tags": ["B-PER", "I-PER", "O", "B-LOC", "O", "O", "O", "O", "O", "O"]
        },
        {
            "id": "hin_7",
            "text": "छठ पूजा का त्योहार पटना और वाराणसी में धूमधाम से मनाया जाता है।",
            "dataset": "Cultural & Historical Texts",
            "description": "Major festival & dual location (Patna, Varanasi)",
            "tokens": ["छठ", "पूजा", "का", "त्योहार", "पटना", "और", "वाराणसी", "में", "धूमधाम", "से", "मनाया", "जाता", "है।"],
            "tags": ["B-MISC", "I-MISC", "O", "O", "B-LOC", "O", "B-LOC", "O", "O", "O", "O", "O", "O"]
        }
    ],

    "Marathi": [
        {
            "id": "mar_1",
            "text": "छत्रपती शिवाजी महाराज यांनी रायगडवर स्वराज्याची स्थापना केली।",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Historic ruler & Raigad fort location",
            "tokens": ["छत्रपती", "शिवाजी", "महाराज", "यांनी", "रायगडवर", "स्वराज्याची", "स्थापना", "केली।"],
            "tags": ["B-PER", "I-PER", "I-PER", "O", "B-LOC", "O", "O", "O"]
        },
        {
            "id": "mar_2",
            "text": "पुणे विद्यापीठ हे महाराष्ट्रातील प्रमुख शैक्षणिक केंद्र आहे।",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "University organization & state location with suffix 'महाराष्ट्रातील'",
            "tokens": ["पुणे", "विद्यापीठ", "हे", "महाराष्ट्रातील", "प्रमुख", "शैक्षणिक", "केंद्र", "आहे।"],
            "tags": ["B-ORG", "I-ORG", "O", "B-LOC", "O", "O", "O", "O"]
        },
        {
            "id": "mar_3",
            "text": "भाभा परमाणु संशोधन केंद्र मुंबईमध्ये स्थित आहे।",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Research organization & Mumbai postposition 'मुंबईमध्ये'",
            "tokens": ["भाभा", "परमाणु", "संशोधन", "केंद्र", "मुंबईमध्ये", "स्थित", "आहे।"],
            "tags": ["B-ORG", "I-ORG", "I-ORG", "I-ORG", "B-LOC", "O", "O"]
        },
        {
            "id": "mar_4",
            "text": "संत ज्ञानेश्वर महाराजांनी आळंदी येथे ज्ञानेश्वरी ग्रंथाची रचना केली।",
            "dataset": "CoNLL-2003 Standard",
            "description": "Saint poet, Alandi location & literature work",
            "tokens": ["संत", "ज्ञानेश्वर", "महाराजांनी", "आळंदी", "येथे", "ज्ञानेश्वरी", "ग्रंथाची", "रचना", "केली।"],
            "tags": ["B-PER", "I-PER", "O", "B-LOC", "O", "B-MISC", "O", "O", "O"]
        },
        {
            "id": "mar_5",
            "text": "लोकमान्य टिळक यांनी २६ जानेवारी रोजी पुण्यात भाषण दिले।",
            "dataset": "CoNLL-2003 Standard",
            "description": "National leader, date expression & Pune city",
            "tokens": ["लोकमान्य", "टिळक", "यांनी", "२६", "जानेवारी", "रोजी", "पुण्यात", "भाषण", "दिले।"],
            "tags": ["B-PER", "I-PER", "O", "B-DATE", "I-DATE", "O", "B-LOC", "O", "O"]
        },
        {
            "id": "mar_6",
            "text": "छत्रपती शिवाजी महाराज यांनी रायगडवर स्वराज्याची स्थापना केली आणि ज्ञानेश्वर महाराजांनी पुण्यात ग्रंथ लिहिला.",
            "dataset": "Synthetic Agglutinative Corpus",
            "description": "Multi-entity sentence with dual PER and LOC spans",
            "tokens": ["छत्रपती", "शिवाजी", "महाराज", "यांनी", "रायगडवर", "स्वराज्याची", "स्थापना", "केली", "आणि", "ज्ञानेश्वर", "महाराजांनी", "पुण्यात", "ग्रंथ", "लिहिला."],
            "tags": ["B-PER", "I-PER", "I-PER", "O", "B-LOC", "O", "O", "O", "O", "B-PER", "O", "B-LOC", "O", "O"]
        },
        {
            "id": "mar_7",
            "text": "गणेशोत्सव हा सण मुंबईत आणि नागपूरमध्ये उत्साहाने साजरा होतो।",
            "dataset": "Cultural & Historical Texts",
            "description": "Major festival & dual locations (Mumbai, Nagpur)",
            "tokens": ["गणेशोत्सव", "हा", "सण", "मुंबईत", "आणि", "नागपूरमध्ये", "उत्साहाने", "साजरा", "होतो।"],
            "tags": ["B-MISC", "O", "O", "B-LOC", "O", "B-LOC", "O", "O", "O"]
        }
    ],

    "Bengali": [
        {
            "id": "ben_1",
            "text": "রবীন্দ্রনাথ ঠাকুর শান্তিনিকেতনে বিশ্বভারতী বিশ্ববিদ্যালয় প্রতিষ্ঠা করেছিলেন।",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Nobel laureate, Santiniketan location & Visva-Bharati University",
            "tokens": ["রবীন্দ্রনাথ", "ঠাকুর", "শান্তিনিকেতনে", "বিশ্বভারতী", "বিশ্ববিদ্যালয়", "প্রতিষ্ঠা", "করেছিলেন।"],
            "tags": ["B-PER", "I-PER", "B-LOC", "B-ORG", "I-ORG", "O", "O"]
        },
        {
            "id": "ben_2",
            "text": "সত্যজিৎ রায় কলকাতায় জন্মগ্রহণ করেন এবং যাদবপুর বিশ্ববিদ্যালয়ে পড়েছিলেন।",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Filmmaker, Kolkata location & Jadavpur University",
            "tokens": ["সত্যজিৎ", "রায়", "কলকাতায়", "জন্মগ্রহণ", "করেন", "এবং", "যাদবপুর", "বিশ্ববিদ্যালয়ে", "পড়েছিলেন।"],
            "tags": ["B-PER", "I-PER", "B-LOC", "O", "O", "O", "B-ORG", "I-ORG", "O"]
        },
        {
            "id": "ben_3",
            "text": "কলকাতা হাইকোর্ট ভারতে ১৮৬২ সালে তৈরি হয়েছিল।",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Court institution, country location & year tag",
            "tokens": ["কলকাতা", "হাইকোর্ট", "ভারতে", "১৮৬২", "সালে", "তৈরি", "হয়েছিল।"],
            "tags": ["B-ORG", "I-ORG", "B-LOC", "B-DATE", "I-DATE", "O", "O"]
        },
        {
            "id": "ben_4",
            "text": "সুভাষচন্দ্র বসু নেতাজি নামে বিখ্যাত এবং তাঁর জন্ম কটকে।",
            "dataset": "CoNLL-2003 Standard",
            "description": "Netaji Subhas Chandra Bose & Cuttack birthplace",
            "tokens": ["সুভাষচন্দ্র", "বসু", "নেতাজি", "নামে", "বিখ্যাত", "এবং", "তাঁর", "জন্ম", "কটকে।"],
            "tags": ["B-PER", "I-PER", "O", "O", "O", "O", "O", "O", "B-LOC"]
        },
        {
            "id": "ben_5",
            "text": "রবীন্দ্রনাথ ঠাকুর ১৯১৩ সালে নোবেল পুরস্কার পান এবং শান্তিনিকেতনে থাকতেন।",
            "dataset": "Synthetic Agglutinative Corpus",
            "description": "Nobel prize award, date & Santiniketan location",
            "tokens": ["রবীন্দ্রনাথ", "ঠাকুর", "১৯১৩", "সালে", "নোবেল", "পুরস্কার", "পান", "এবং", "শান্তিনিকেতনে", "থাকতেন।"],
            "tags": ["B-PER", "I-PER", "B-DATE", "I-DATE", "B-MISC", "I-MISC", "O", "O", "B-LOC", "O"]
        },
        {
            "id": "ben_6",
            "text": "দুর্গাপূজা পশ্চিমবঙ্গে মহা সমারোহে পালিত হয়।",
            "dataset": "Cultural & Historical Texts",
            "description": "Durga Puja festival in West Bengal",
            "tokens": ["দুর্গাপূজা", "পশ্চিমবঙ্গে", "মহা", "সমারোহে", "পালিত", "হয়।"],
            "tags": ["B-MISC", "B-LOC", "O", "O", "O", "O"]
        }
    ],

    "Gujarati": [
        {
            "id": "guj_1",
            "text": "મહાત્મા ગાંધીનો જન્મ પોરબંદરમાં થયો હતો.",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Mahatma Gandhi & Porbandar birthplace",
            "tokens": ["મહાત્મા", "ગાંધીનો", "જન્મ", "પોરબંદરમાં", "થયો", "હતો."],
            "tags": ["B-PER", "I-PER", "O", "B-LOC", "O", "O"]
        },
        {
            "id": "guj_2",
            "text": "ગુજરાત વિદ્યાપીઠ અમદાવાદમાં આવેલી છે.",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Gujarat Vidyapith in Ahmedabad",
            "tokens": ["ગુજરાત", "વિદ્યાપીઠ", "અમદાવાદમાં", "આવેલી", "છે."],
            "tags": ["B-ORG", "I-ORG", "B-LOC", "O", "O"]
        },
        {
            "id": "guj_3",
            "text": "સરદાર વલ્લભભાઈ પટેલનો જન્મ નડિયાદમાં થયો હતો.",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Sardar Vallabhbhai Patel & Nadiad city",
            "tokens": ["સરદાર", "વલ્લભભાઈ", "પટેલનો", "જન્મ", "નડિયાદમાં", "થયો", "હતો."],
            "tags": ["B-PER", "I-PER", "I-PER", "O", "B-LOC", "O", "O"]
        },
        {
            "id": "guj_4",
            "text": "ઈસરો અમદાવાદ સેન્ટર દ્વારા નવા ઉપગ્રહનું પરીક્ષણ કરાયું.",
            "dataset": "CoNLL-2003 Standard",
            "description": "ISRO space center in Ahmedabad",
            "tokens": ["ઈસરો", "અમદાવાદ", "સેન્ટર", "દ્વારા", "નવા", "ઉપગ્રહનું", "પરીક્ષણ", "કરાયું."],
            "tags": ["B-ORG", "I-ORG", "I-ORG", "O", "O", "O", "O", "O"]
        },
        {
            "id": "guj_5",
            "text": "મહાત્મા ગાંધી અને સરદાર પટેલે સ્વતંત્રતા સંગ્રામમાં મહત્વનો ભાગ ભજવ્યો.",
            "dataset": "Synthetic Agglutinative Corpus",
            "description": "Dual leaders in freedom movement",
            "tokens": ["મહાત્મા", "ગાંધી", "અને", "સરદાર", "પટેલે", "સ્વતંત્રતા", "સંગ્રામમાં", "મહત્વનો", "ભાગ", "ભજવ્યો."],
            "tags": ["B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "O", "O", "O", "O"]
        },
        {
            "id": "guj_6",
            "text": "જન્માષ્ટમી ઉત્સવ દ્વારકામાં ખૂબ જ ધામધૂમથી ઉજવાય છે.",
            "dataset": "Cultural & Historical Texts",
            "description": "Janmashtami festival in Dwarka city",
            "tokens": ["જન્માષ્ટમી", "ઉત્સવ", "દ્વારકામાં", "ખૂબ", "જ", "ધામધૂમથી", "ઉજવાય", "છે."],
            "tags": ["B-MISC", "I-MISC", "B-LOC", "O", "O", "O", "O", "O"]
        }
    ],

    "Odia": [
        {
            "id": "odi_1",
            "text": "ଉତ୍କଳମଣି ଗୋପବନ୍ଧୁ ଦାସ କଟକରେ ଜନ୍ମଗ୍ରହଣ କରିଥିଲେ।",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Gopabandhu Das & Cuttack birthplace",
            "tokens": ["ଉତ୍କଳମଣି", "ଗୋପବନ୍ଧୁ", "ଦାସ", "କଟକରେ", "ଜନ୍ମଗ୍ରହଣ", "କରିଥିଲେ।"],
            "tags": ["O", "B-PER", "I-PER", "B-LOC", "O", "O"]
        },
        {
            "id": "odi_2",
            "text": "ଉତ୍କଳ ବିଶ୍ୱବିଦ୍ୟାଳୟ ଭୁବନେଶ୍ୱରରେ ଅବସ୍ଥିତ।",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Utkal University in Bhubaneswar capital",
            "tokens": ["ଉତ୍କଳ", "ବିଶ୍ୱବିଦ୍ୟାଳୟ", "ଭୁବନେଶ୍ୱରରେ", "ଅବସ୍ଥିତ।"],
            "tags": ["B-ORG", "I-ORG", "B-LOC", "O"]
        },
        {
            "id": "odi_3",
            "text": "ବୀର ସୁରେନ୍ଦ୍ର ସାଏ ସମ୍ବଲପୁରେ ଆନ୍ଦୋଳନ କରିଥିଲେ।",
            "dataset": "CoNLL-2003 Standard",
            "description": "Freedom fighter Surendra Sai in Sambalpur",
            "tokens": ["ବୀର", "ସୁରେନ୍ଦ୍ର", "ସାଏ", "ସମ୍ବଲପୁରେ", "ଆନ୍ଦୋଳନ", "କରିଥିଲେ।"],
            "tags": ["B-PER", "I-PER", "I-PER", "B-LOC", "O", "O"]
        },
        {
            "id": "odi_4",
            "text": "ରଥଯାତ୍ରା ପୁରୀରେ ଆଷାଢ଼ ମାସରେ ମନାଯାଏ।",
            "dataset": "Synthetic Agglutinative Corpus",
            "description": "Ratha Yatra festival in Puri & month date tag",
            "tokens": ["ରଥଯାତ୍ରା", "ପୁରୀରେ", "ଆଷାଢ଼", "ମାସରେ", "ମନାଯାଏ।"],
            "tags": ["B-MISC", "B-LOC", "B-DATE", "I-DATE", "O"]
        },
        {
            "id": "odi_5",
            "text": "ନନ୍ଦନକାନନ ଜାତୀୟ ଉଦ୍ୟାନ ଭୁବନେଶ୍ୱରରେ ଅଛି।",
            "dataset": "Cultural & Historical Texts",
            "description": "Nandankanan National Park in Bhubaneswar",
            "tokens": ["ନନ୍ଦନକାନନ", "ଜାତୀୟ", "ଉଦ୍ୟାନ", "ଭୁବନେଶ୍ୱରରେ", "ଅଛି।"],
            "tags": ["B-LOC", "I-LOC", "I-LOC", "B-LOC", "O"]
        }
    ],

    "Punjabi": [
        {
            "id": "pun_1",
            "text": "ਗੁਰੂ ਨਾਨਕ ਦੇਵ ਜੀ ਦਾ ਜਨਮ ਸੁਲਤਾਨਪੁਰ ਵਿੱਚ ਹੋਇਆ ਸੀ।",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Guru Nanak Dev Ji & Sultanpur town",
            "tokens": ["ਗੁਰੂ", "ਨਾਨਕ", "ਦੇਵ", "ਜੀ", "ਦਾ", "ਜਨਮ", "ਸੁਲਤਾਨਪੁਰ", "ਵਿੱਚ", "ਹੋਇਆ", "ਸੀ।"],
            "tags": ["B-PER", "I-PER", "I-PER", "O", "O", "O", "B-LOC", "O", "O", "O"]
        },
        {
            "id": "pun_2",
            "text": "ਪੰਜਾਬ ਯੂਨੀਵਰਸਿਟੀ ਚੰਡੀਗੜ੍ਹ ਵਿੱਚ ਸਥਿਤ ਹੈ।",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Panjab University in Chandigarh",
            "tokens": ["ਪੰਜਾਬ", "ਯੂਨੀਵਰਸਿਟੀ", "ਚੰਡੀਗੜ੍ਹ", "ਵਿੱਚ", "ਸਥਿਤ", "ਹੈ।"],
            "tags": ["B-ORG", "I-ORG", "B-LOC", "O", "O", "O"]
        },
        {
            "id": "pun_3",
            "text": "ਭਗਤ ਸਿੰਘ ਨੇ 1929 ਵਿੱਚ ਲਾਹੌਰ ਵਿੱਚ ਆਵਾਜ਼ ਉਠਾਈ ਸੀ।",
            "dataset": "CoNLL-2003 Standard",
            "description": "Bhagat Singh, year 1929 & Lahore city",
            "tokens": ["ਭਗਤ", "ਸਿੰਘ", "ਨੇ", "1929", "ਵਿੱਚ", "ਲਾਹੌਰ", "ਵਿੱਚ", "ਆਵਾਜ਼", "ਉਠਾਈ", "ਸੀ।"],
            "tags": ["B-PER", "I-PER", "O", "B-DATE", "O", "B-LOC", "O", "O", "O", "O"]
        },
        {
            "id": "pun_4",
            "text": "ਵਿਸਾਖੀ ਦਾ ਤਿਉਹਾਰ ਅੰਮ੍ਰਿਤਸਰ ਵਿੱਚ 13 ਅਪ੍ਰੈਲ ਨੂੰ ਮਨਾਇਆ ਜਾਂਦਾ ਹੈ।",
            "dataset": "Synthetic Agglutinative Corpus",
            "description": "Vaisakhi festival, Amritsar location & date tag",
            "tokens": ["ਵਿਸਾਖੀ", "ਦਾ", "ਤਿਉਹਾਰ", "ਅੰਮ੍ਰਿਤਸਰ", "ਵਿੱਚ", "13", "ਅਪ੍ਰੈਲ", "ਨੂੰ", "ਮਨਾਇਆ", "ਜਾਂਦਾ", "ਹੈ।"],
            "tags": ["B-MISC", "O", "O", "B-LOC", "O", "B-DATE", "I-DATE", "O", "O", "O", "O"]
        },
        {
            "id": "pun_5",
            "text": "ਗੋਲਡਨ ਟੈਂਪਲ ਅੰਮ੍ਰਿਤਸਰ ਵਿੱਚ ਪ੍ਰਸਿੱਧ ਧਾਰਮਿਕ ਅਸਥਾਨ ਹੈ।",
            "dataset": "Cultural & Historical Texts",
            "description": "Golden Temple shrine in Amritsar",
            "tokens": ["ਗੋਲਡਨ", "ਟੈਂਪਲ", "ਅੰਮ੍ਰਿਤਸਰ", "ਵਿੱਚ", "ਪ੍ਰਸਿੱਧ", "ਧਾਰਮਿਕ", "ਅਸਥਾਨ", "ਹੈ।"],
            "tags": ["B-LOC", "I-LOC", "B-LOC", "O", "O", "O", "O", "O"]
        }
    ],

    "Other / English": [
        {
            "id": "eng_1",
            "text": "Dr. Rajendra Prasad was born in Jiradei village of Bihar.",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "English translation: Dr. Rajendra Prasad & Jiradei, Bihar",
            "tokens": ["Dr.", "Rajendra", "Prasad", "was", "born", "in", "Jiradei", "village", "of", "Bihar."],
            "tags": ["O", "B-PER", "I-PER", "O", "O", "O", "B-LOC", "O", "O", "B-LOC"]
        },
        {
            "id": "eng_2",
            "text": "Indian Space Research Organisation (ISRO) is headquartered in Bengaluru.",
            "dataset": "IndicNER / WikiANN (Real)",
            "description": "Space agency ISRO & Bengaluru city",
            "tokens": ["Indian", "Space", "Research", "Organisation", "(ISRO)", "is", "headquartered", "in", "Bengaluru."],
            "tags": ["B-ORG", "I-ORG", "I-ORG", "I-ORG", "B-ORG", "O", "O", "O", "B-LOC"]
        },
        {
            "id": "eng_3",
            "text": "Birsa Munda led the Ulgulan movement in Ranchi during 1900.",
            "dataset": "CoNLL-2003 Standard",
            "description": "Birsa Munda, Ulgulan movement, Ranchi & year 1900",
            "tokens": ["Birsa", "Munda", "led", "the", "Ulgulan", "movement", "in", "Ranchi", "during", "1900."],
            "tags": ["B-PER", "I-PER", "O", "O", "B-MISC", "I-MISC", "O", "B-LOC", "O", "B-DATE"]
        },
        {
            "id": "eng_4",
            "text": "Chhatrapati Shivaji Maharaj established Swarajya at Raigad in Maharashtra.",
            "dataset": "Synthetic Agglutinative Corpus",
            "description": "Ruler, Raigad fort & Maharashtra state",
            "tokens": ["Chhatrapati", "Shivaji", "Maharaj", "established", "Swarajya", "at", "Raigad", "in", "Maharashtra."],
            "tags": ["B-PER", "I-PER", "I-PER", "O", "O", "O", "B-LOC", "O", "B-LOC"]
        },
        {
            "id": "eng_5",
            "text": "Chhath Puja is celebrated with great fervor in Patna and Varanasi.",
            "dataset": "Cultural & Historical Texts",
            "description": "Chhath Puja festival in Patna & Varanasi",
            "tokens": ["Chhath", "Puja", "is", "celebrated", "with", "great", "fervor", "in", "Patna", "and", "Varanasi."],
            "tags": ["B-MISC", "I-MISC", "O", "O", "O", "O", "O", "O", "B-LOC", "O", "B-LOC"]
        }
    ]
}


def get_examples_for_lang_and_dataset(language: str, dataset_category: str = "All Datasets") -> List[Dict[str, Any]]:
    """
    Retrieves examples matching specified target language and dataset filter.
    Falls back gracefully if language or dataset is not found.
    """
    lang_examples = MULTILINGUAL_EXAMPLES.get(language, MULTILINGUAL_EXAMPLES.get("Bhojpuri", []))
    
    if dataset_category == "All Datasets":
        return lang_examples
    
    filtered = [ex for ex in lang_examples if ex["dataset"] == dataset_category]
    if not filtered:
        # Fallback to all examples for that language
        return lang_examples
    return filtered
