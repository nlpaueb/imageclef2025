from medcat.cat import CAT


class MedCatScorer:
    def __init__(self, model_path, semantic_types=None):
        self.cat = CAT.load_model_pack(model_path)
        if semantic_types:
            type_ids_filter = set(semantic_types)
        else:
            type_ids_filter = {
                "T048",
                "T197",
                "T088",
                "T055",
                "T029",
                "T004",
                "T043",
                "T101",
                "T129",
                "T069",
                "T045",
                "T079",
                "T167",
                "T049",
                "T010",
                "T080",
                "T121",
                "T082",
                "T066",
                "T040",
                "T170",
                "T086",
                "T058",
                "T130",
                "T195",
                "T109",
                "T127",
                "T037",
                "T125",
                "T081",
                "T071",
                "T061",
                "T126",
                "T192",
                "T077",
                "T073",
                "T168",
                "T185",
                "T089",
                "T074",
                "T001",
                "T059",
                "T104",
                "T083",
                "T051",
                "T044",
                "T002",
                "T194",
                "T028",
                "T094",
                "T057",
                "T053",
                "T090",
                "T060",
                "T056",
                "T201",
                "T171",
                "T013",
                "T190",
                "T085",
                "T087",
                "T100",
                "T042",
                "T203",
                "T031",
                "T078",
                "T047",
                "T091",
                "T052",
                "T021",
                "T017",
                "T064",
                "T103",
                "T018",
                "T098",
                "T011",
                "T116",
                "T200",
                "T012",
                "T099",
                "T008",
                "T072",
                "T041",
                "T014",
                "T030",
                "T204",
                "T016",
                "T032",
                "T020",
                "T096",
                "T097",
                "T005",
                "T120",
                "T038",
                "T191",
                "T093",
                "T092",
                "T007",
                "T019",
                "T025",
                "T122",
                "T075",
                "T046",
                "T039",
                "T065",
                "T015",
                "T114",
                "T054",
                "T095",
                "T068",
                "T063",
                "T062",
                "T102",
                "T070",
                "T033",
                "T050",
                "T184",
                "T123",
                "T034",
                "T026",
                "T169",
                "T067",
                "T024",
                "T131",
                "T196",
                "T022",
                "T023",
            } # MEDCON types
        cui_filters = set()
        for type_ids in type_ids_filter:
            cui_filters.update(self.cat.cdb.addl_info["type_id2cuis"][type_ids])
        self.cat.cdb.config.linking["filters"]["cuis"] = cui_filters

    def get_matches(self, text):
        concepts = {}
        cui_list = []

        entities = self.cat.get_entities(text)["entities"]
        for ent in entities.values():  # Fix: iterate over the values of the entities dictionary
            term = ent["pretty_name"]
            cui = ent["cui"]
            if cui not in concepts.get(term, []):
                concepts.setdefault(term, []).append(cui)
                cui_list.append(cui)
        return concepts, cui_list

    def score(self, reference, prediction):
        true_concept, true_cuis = self.get_matches(reference)
        pred_concept, pred_cuis = self.get_matches(prediction)

        try:
            # Count correctly predicted CUIs
            correct_predictions = sum(
                1
                for key in true_concept
                for cui in true_concept[key]
                if cui in pred_cuis
            )

            # Calculate precision, recall, and F1 score
            precision = correct_predictions / len(pred_cuis) if pred_cuis else 0
            recall = correct_predictions / len(true_cuis) if true_cuis else 0
            F1 = (
                2 * (precision * recall) / (precision + recall)
                if precision + recall > 0
                else 0
            )

            return F1
        except Exception as e:
            print(f"Error calculating F1 score: {e}")
            return 0


if __name__ == "__main__":
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models/MedCAT/umls_self_train_model_pt2ch_3760d588371755d0.zip")
    
    if not os.path.exists(model_path):
        raise Exception("MedCAT model not found at {}".format(model_path))
    
    scorer = MedCatScorer(model_path=model_path)
    
    reference = "The patient was diagnosed with pneumonia. But the patient also has a history of asthma."
    prediction = "The patient has pneumonia."
    
    score = scorer.score(reference, prediction)
    print(f"MedCat Score: {score}")