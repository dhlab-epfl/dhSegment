import json
import os


def _parse_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


class ExperimentResult:
    def __init__(self, base_post_process_folder: str, base_key: str):
        self.base_post_process_folder = base_post_process_folder
        validation_filename = os.path.join(base_post_process_folder, 'validation_scores.json')
        self.validation_scores = _parse_json(validation_filename)

        post_process_config_filename = os.path.join(base_post_process_folder, 'post_process_params.json')
        self.post_process_config = _parse_json(post_process_config_filename)
        model_config_filename = os.path.join(os.path.dirname(os.path.dirname(self.base_post_process_folder)),
                                             'config.json')
        self.model_config = _parse_json(model_config_filename)

        self.base_key = base_key

    def get_best_validated_epoch(self):
        best_epoch = sorted(self.validation_scores.values(), key=lambda v: v[self.base_key], reverse=True)[0]
        return best_epoch

    def get_best_validated_score(self, key=None):
        epoch = self.get_best_validated_epoch()
        return epoch[key if key is not None else self.base_key]

    def get_best_model_folder(self):
        epoch = self.get_best_validated_epoch()
        return os.path.join(os.path.dirname(os.path.dirname(self.base_post_process_folder)),
                            'export', str(epoch['timestamp']))

    def __repr__(self):
        return "ExperimentResult: \n\tmodel -> {}\n\tpost-process -> {}\n\tfolder -> {}".format(
            self.model_config, self.post_process_config, self.get_best_model_folder())
