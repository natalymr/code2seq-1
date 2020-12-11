import collections
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from code2seq_dataset.info_classes import CommitLogLine
from common_dataset.logs import COMMON_SEP


def replace_file_number_to_commit_hash(splitted_commits: Path, results_dir: Path) -> None:
    file_type = 'val'
    index_to_commit_hash: Dict[str, str] = {}
    with open(splitted_commits / f'{file_type.upper()}_commits.log', 'r') as commit_log:
        for ind, line in enumerate(commit_log):
            key = '/{:0>6d}.txt'.format(ind)
            line = line.strip('\n')
            index_to_commit_hash[key] = line
    with open(results_dir / f'top1000_results_{file_type.lower()}.txt', 'r') as results_log, \
        open(results_dir / f'results_with_commits_hash_{file_type.lower()}', 'w') as output_f:
        for line in results_log:
            file_number, generated_msg = line.split('\t')
            generated_msg = generated_msg.strip('\n')

            output_f.write(f'{index_to_commit_hash[file_number]}\t{generated_msg}\n')


def concat_all_three_files(results_directory: Path) -> None:
    with open(results_directory / f'results_with_commits_hash_train', 'r') as input_train, \
        open(results_directory / f'results_with_commits_hash_test', 'r') as input_test, \
        open(results_directory / f'results_with_commits_hash_val', 'r') as input_val, \
        open(results_directory / f'results_with_commits_hash_common.txt', 'w') as output_f:

        for line in input_train:
            output_f.write(line)
        for line in input_test:
            output_f.write(line)
        for line in input_val:
            output_f.write(line)


def replace_duplicates(results_directory: Path) -> None:
    def replace_duplicates_str(value: str) -> str:
        result: List[str] = []
        str_in_list = value.split()
        result.append(str_in_list[0])

        for i in range(1, len(str_in_list)):
            current_word = str_in_list[i]
            if current_word != result[-1]:
                result.append(current_word)

        return ' '.join(result)

    with open(results_directory / f'results_with_commits_hash_common.txt', 'r') as input_f, \
        open(results_directory / f'results_with_commits_hash_common_delete_duplicates.txt', 'w') as output_f:

        for line in input_f:
            commit_hash, generated_msg = line.split('\t')
            generated_msg = generated_msg.strip('\n')
            # print(f'from - {generated_msg}, to - {replace_duplicates_str(generated_msg)}')
            output_f.write(f'{commit_hash}\t{replace_duplicates_str(generated_msg)}\n')


def insert_original_message(results_file: Path, top_1000_dir: Path):
    meta_commit_to_generated_message: Dict[str, str] = {}
    meta_commit_to_original_message: Dict[str, str] = {}
    directory_to_its_commits: Dict[str, List[str]] = collections.defaultdict(list)
    with open(results_file, 'r') as input_file:
        for line in input_file:
            meta_commit, generated_message = line.split('\t')
            meta_commit_to_generated_message[meta_commit] = generated_message
            # print(meta_commit)
            user_and_directory, commit = meta_commit[:-41], meta_commit[-40:]
            directory_to_its_commits[f'{user_and_directory}'].append(commit)

    for file in tqdm(top_1000_dir.iterdir()):
        if file.name.endswith('.commits.logs'):
            # print(file.name)
            directory_name = file.name[:-13]
            if directory_name in directory_to_its_commits.keys():
                with open(file, 'r') as log_file:
                    for line in log_file:
                        log_line = CommitLogLine.parse_from_line(line, separator=COMMON_SEP)
                        if log_line.current_commit in directory_to_its_commits[directory_name]:
                            meta_commit_to_original_message[
                                f'{directory_name}_{log_line.current_commit}'] = log_line.message
    refs, pred = [], []
    with open(results_file.parent / 'common_results_orig_generated_duplicates.txt', 'w') as output_file:
        for meta_commit, generated_message in meta_commit_to_generated_message.items():
            refs.append(meta_commit_to_original_message[meta_commit].strip())
            pred.append(generated_message.strip())
            # output_file.write(
            #     f'{meta_commit}\t{generated_message.strip()}\t{meta_commit_to_original_message[meta_commit].strip()}\n')
    return refs, pred


def get_nltk_bleu_score_for_corpora_print(refs: List[str], preds: List[str], results_dir: Path) -> float:
    import nltk
    total_bleu = 0.
    msg_vs_bleu = {}
    for ref, pred in zip(refs, preds):
        if len(pred.split()) == 0:
            bleu = 0
        else:
            bleu = nltk.translate.bleu_score.sentence_bleu([ref.split()], pred.split(),
                                                           auto_reweigh=True,
                                                           smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method7) * 100
        msg_vs_bleu[f'{ref.strip()} || {pred.strip()}'] = bleu
        total_bleu += bleu
    msg_vs_bleu = collections.OrderedDict(sorted(msg_vs_bleu.items(), key=lambda t: t[1], reverse=True))
    with open(results_dir.joinpath('sorted_bleu_duplicates.txt'), 'w') as output_f:
        for msg, bleu in msg_vs_bleu.items():
            if bleu > 0.5:
                output_f.write('{:.3f} || {}\n'.format(bleu, msg))
    print(total_bleu / len(refs))
    return total_bleu / len(refs)


if __name__ == '__main__':
    lca_dir = Path.cwd().parent.parent.parent.parent / 'new_data' / 'processed_data'
    results_dir: Path = lca_dir / 'results_commit2seq_two_input_for_top1000'
    splitted_commits: Path = lca_dir / 'c2s_paths' / 'two_inputs' / 'two_input'
    # replace_file_number_to_commit_hash(splitted_commits, results_dir)
    # concat_all_three_files(results_dir)
    # replace_duplicates(results_dir)
    refs, pred = insert_original_message(results_dir / 'results_with_commits_hash_common_delete_duplicates.txt',
                                         lca_dir.parent / 'raw_data')
    get_nltk_bleu_score_for_corpora_print(refs, pred, results_dir)
