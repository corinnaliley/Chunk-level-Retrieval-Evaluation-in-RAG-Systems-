# -----------------------------------------------------------------------------
# testset.py — Benchmark query set + gold references
#
# Scientific / reproducibility notes
# - This file defines the complete evaluation dataset:
#     * query text
#     * query type (keyword / semantic / multi-hop)
#     * gold reference strings (substring-based ground truth)
# - Evaluation metrics in benchmark.py depend strictly on these gold strings.
#   Any change here changes reported scores.
# - Multi-hop queries require ALL listed gold snippets to appear in retrieved
#   chunks to count as full Hit@k.
# - For paper reporting, freeze this file and record the repository commit hash.
#
# -----------------------------------------------------------------------------

# testset.py


_TESTSET = {

    # KEYWORD QUERIES (exakte Begriffe, Zahlen, Listen)



    "abroad_forms": {
        "query": "Was muss vor Antritt eines Studienaufenthalts im Ausland abgeschlossen werden?",
        "type": "keyword",
        "gold": (
            "„Learning Agreement“ nach § 13 Abs. 3 abgeschlossen werden"
        ),
    },

    "aptitude_test": {
        "query": "Wieviele Punkte muss die Bewerberin oder der Bewerber mindestens erreicht haben, um den Eignungstest für den Informatik-Master zu bestehen?",
        "type": "keyword",
        "gold": (
            "mindestens 60 Prozent der Punkte erreicht"
        ),
    },

    "bio_bachelor_thesis_requirements": {
        "query": "Welche Voraussetzungen gelten für die Zulassung zur Bachelorarbeit im Fach Biologie?",
        "type": "keyword",
        "gold": (
            "Es müssen mindestens 53 Credits im Fach Biologie absolviert worden sein, darunter alle Pflichtmodule im Umfang von insgesamt 43 C."
        ),
    },

    "bio_orientation_modules_rule": {
        "query": "Wie viele Orientierungsmodule müssen im Fach Biologie erfolgreich abgeschlossen sein?",
        "type": "keyword",
        "gold": (
            "Der erfolgreiche Abschluss von drei der vier Orientierungsmodule"
        ),
    },

    "physik_orientation_modules": {
        "query": "Welche Module sind im Studienfach Physik als Orientierungsmodule festgelegt?",
        "type": "keyword",
        "gold": (
            "Die Module B.Phy.101 und B.phy.102 sind Orientierungsmodule."
        ),
    },

    "submission_deadline_definition": {
        "query": "Wann gilt ein Zulassungsantrag als fristgerecht eingegangen?",
        "type": "keyword",
        "gold": (
            "bei der Universität eingegangen sein"
        ),
    },

    "bio_exam_registration_type": {
        "query": "Welche Vorkenntnisse werden im Fach Biologie erwartet?",
        "type": "keyword",
        "gold": (
            "gute naturwissenschaftliche Grundkenntnisse in Mathematik, Chemie, Physik und Biologie "
        ),
    },

    "physik_grade_improvement": {
        "query": "Wie oft dürfen bestandene Modulprüfungen im Fach Physik zur Notenverbesserung wiederholt werden?",
        "type": "keyword",
        "gold": (
            "bis zu zwei bestandene Modulprüfungen"
        ),
    },

    "lehramt_profile_credits": {
        "query": "Wie viele Credits umfasst das lehramtbezogene Profil im Professionalisierungsbereich?",
        "type": "keyword",
        "gold": (
            "36 C"
        ),
    },


    # SEMANTIC QUERIES (Paraphrase, Erklärung, Kontext)


    "paper_application": {
        "query": "Kann das Bewerbungsverfahren auf Papier durchgeführt werden?",
        "type": "semantic",
        "gold": (
            "Das Bewerbungsverfahren wird im nachfolgenden Umfang als elektronisches Verfahren durchgeführt."
        ),
    },

    "second_degree": {
        "query": "Was benötigt man für die Bewerbung um ein Zweitstudium?",
        "type": "semantic",
        "gold": (
            "ein Scan des Zeugnisses des erfolgreich abgeschlossenen Erststudiums sowie eine ausführliche Darlegung, aus welchen Gründen ein Zweitstudium angestrebt wird"
        ),
    },

    "master_degrees": {
        "query": "Welche Master-Abschlüsse gibt es?",
        "type": "semantic",
        "gold": [
            "„Master of Arts“ (abgekürzt: „M.A.“)",
            "„Master of Science“ (abgekürzt: „M.Sc.“)",
            "„Master of Education” (abgekürzt: „M.Ed.“)",
            "„Master of Laws” (abgekürzt: „LL.M.”)",
        ],
    },

    "ects": {
        "query": "Für wieviel Zeit steht ein ETCS?",
        "type": "semantic",
        "gold": (
            "Arbeitsaufwand von 30 Zeitstunden"
        ),
    },

    "part_time": {
        "query": "Wieviele Credits im Semester dürfen höchstens bzw. müssen mindestens im Teilzeitstudium erbracht werden?",
        "type": "semantic",
        "gold": [
            "nicht weniger als ein Drittel (10 C je Semester)",
            "nicht mehr als fünf Sechstel (25 C je Semester)",
        ],
    },

    "english_level": {
        "query": "Wie können Bewerber des Masterstudiengangs Informatik ein ausreichendes Englisch-Level nachweisen?",
        "type": "semantic",
        "gold": (
            "Niveau B2 oder höher nach dem Gemeinsamen europäischen Referenzrahmen für Sprachen"
        ),
    },

    "ombuds": {
        "query": "Wen kann man Fragen zu guter wissenschaftlicher Praxis stellen?",
        "type": "semantic",
        "gold": (
            "Ombudsgremium"
        ),
    },

    "take_home": {
        "query": "Dürfen Kommilitonen eine Take-Home-Klausur zusammen machen?",
        "type": "semantic",
        "gold": (
            "Teilnehmer*innen müssen in\n"
            "Textform erklären, dass sie die THK selbstständig ohne Hilfe Dritter oder Verwendung unzulässiger\n"
            "Hilfsmittel bearbeitet haben."
        ),
    },

    "physik_study_goals_semantic": {
        "query": "Welche fachlichen Kompetenzen sollen Absolventinnen und Absolventen im Studienfach Physik erwerben?",
        "type": "semantic",
        "gold": (
            "Sie sollen befähigt sein, verschiedene Teilgebiete der Physik durch das Verständnis wichtiger "
            "gemeinsamer Konzepte zu verknüpfen und sich aktuelle Fragestellungen physikalischer Forschung "
            "selbstständig erarbeiten können."
        ),
    },

    "cross_zfb_biologie_study_structure": {
        "query": "Wie ist das Studium im Zwei-Fächer-Bachelor aufgebaut und wie viele Credits entfallen dabei auf das Fach Biologie?",
        "type": "semantic",
        "gold": [
            "Das  Studium  umfasst  180  Anrechnungspunkte",
            "auf jedes der beiden gewählten Studienfächer jeweils 66 C (Fachstudium; Kerncurriculum)"
        ],
    },


    # =================================================
    # MULTI-HOP QUERIES (mehrere Textstellen / Dokumente)
    # =================================================

    "deadlines": {
        "query": "Wie gelten die Fristen für den Zulassungsantrag mit und ohne Sonderquote?",
        "type": "multi-hop",
        "gold": [
            "für das Wintersemester bis zum 15. Juli,",
            "für das Sommersemester bis zum 15. Januar",
            "Sonderquote nach § 22 Abs. 1 Satz 1 Nr1 Hochschulzulassungsverordnung (Ausländerquote)",
            "für das Wintersemester bis zum 30. April eines Jahres,",
            "für das Sommersemester bis zum 31. Oktober des Vorjahres",
        ],
    },

    "thesis": {
        "query": "Wieviele Credits muss man haben, um zur Informatik-Bachelorarbeit und wieviele, um zur Informatik-Masterarbeit zugelassen zu werden?",
        "type": "multi-hop",
        "gold": [
            "Umfang von mindestens 83 C",
            "Umfang von wenigstens 48 C",
        ],
    },

    "cross_bio_physik_bachelor_thesis_requirements": {
        "query": "Welche Voraussetzungen gelten für die Zulassung zur Bachelorarbeit in den Studienfächern Biologie und Physik?",
        "type": "multi-hop",
        "gold": [
            "Es müssen mindestens 53 Credits im Fach Biologie absolviert worden sein, darunter alle Pflichtmodule im Umfang von insgesamt 43 C.",
            "Voraussetzung für die Zulassung zur Bachelor-Arbeit im Studienfach „Physik“ ist der Nachweis von 48 C aus dem Kerncurriculum.",
        ],
    },

    "cross_lehramt_biologie_modules": {
        "query": "Welche fach-spezifischen Module müssen im Lehramtbezogenen Profil absolviert werden und welches biologische Modul erfüllt die fachdidaktische Kompetenz?",
        "type": "multi-hop",
        "gold": [
            "beiden Studienfächern das jeweils in der Modulübersicht gesondert ausgewiesene Modul zur fachdidaktischen Kompetenz",
            "Einführung in die Didaktik der Biologie",
        ],
    },

    "cross_zfb_physik_orientation_modules": {
        "query": "Was sind Orientierungsmodule im Zwei-Fächer-Bachelor und welche Module gelten im Fach Physik als Orientierungsmodule?",
        "type": "multi-hop",
        "gold": (
            "B.Phy.101 und B.Phy.102 sind Orientierungsmodule."
        ),
    },
}


# abgeleitete Pipeline-Strukturen

TEST_QUERIES = [
    (k, v["query"]) for k, v in _TESTSET.items()
]


REFERENCE_ANSWERS = {
    k: v["gold"] for k, v in _TESTSET.items()
}

QUERY_META = {
    k: {
        "id": k,
        "type": v["type"],
    }
    for k, v in _TESTSET.items()
}