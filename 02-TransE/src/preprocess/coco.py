import os
import argparse
import json

import pandas as pd

from easydict import EasyDict as edict

from collections import defaultdict, Counter


def read_courses(config):
    # Read course_latest.csv
    course_latest_file = os.path.join(config.original_data_dir, "course_latest.csv")
    course_latest = pd.read_csv(course_latest_file, encoding="utf-8")

    # Remove courses not in english
    course_latest = course_latest[
        (course_latest.language == "english")
        & (course_latest.second_level_category != "other")
    ]
    return course_latest


def read_teach(config, course_latest):
    # Read teach_latest.csv
    teach_latest_file = os.path.join(config.original_data_dir, "teach_latest.csv")
    teach_latest = pd.read_csv(teach_latest_file, encoding="utf-8")

    # Filter out courses not in course_latest
    teach_latest = teach_latest[
        teach_latest.course_id.isin(course_latest.course_id.unique())
    ]

    return teach_latest


def get_instructors_ids(teach_latest):
    # Filter out instructors with less than 2 courses
    instructors_min_two_courses = teach_latest[
        teach_latest.groupby("instructor_id")["instructor_id"].transform("size") > 2
    ]

    # Get the course_ids of the instructors with more than 2 courses
    instructor_ids = instructors_min_two_courses.instructor_id.unique()

    return instructor_ids


def get_valid_courses(teach_latest, instructor_ids, course_latest):
    # Filter out courses with instructors with less than 2 courses
    courses_of_valid_instructors = teach_latest[
        teach_latest.instructor_id.isin(instructor_ids)
    ]

    # Get the course_ids of the courses with instructors with more than 2 courses
    course_ids = courses_of_valid_instructors.course_id.unique()

    # Filter out courses with instructors with less than 2 courses
    valid_courses = course_latest[(course_latest.course_id.isin(course_ids))]

    return valid_courses


def get_student_enrolments(config, valid_courses):
    # Read evaluate_latest.csv
    evaluate_latest = pd.read_csv(
        os.path.join(config.original_data_dir, "evaluate_latest.csv"),
        encoding="utf-8",
        index_col=0,
        # low_memory=False,
    )
    # Filter out courses not in valid_courses
    student_enrolments = evaluate_latest[
        evaluate_latest.course_id.isin(valid_courses.course_id)
    ]

    # Filter out students with less than 2 courses
    student_enrolments = student_enrolments[
        student_enrolments.groupby("learner_id")["learner_id"].transform("size")
        >= config.min_user_count
    ]
    return student_enrolments


def get_course_skill(config):
    # Read course_skill.csv
    course_skills_file = os.path.join(config.original_data_dir, "course_skill.csv")
    course_skills = pd.read_csv(course_skills_file, encoding="utf-8")
    return course_skills


def write_entity(config, entity_ids, entity_name):
    entity_ids_id_to_num = {}
    instructor_file = os.path.join(config.processed_data_dir, entity_name + ".txt")
    with open(instructor_file, "w", encoding="utf-8") as f:
        for i, instr in enumerate(entity_ids):
            f.write(str(instr) + "\n")
            entity_ids_id_to_num[instr] = i
    return entity_ids_id_to_num


def get_course_info(valid_courses, course_latest, teach_latest, instructor_ids):
    courses_info = {}
    course_num_to_id = {}
    instructor_list = list(instructor_ids)

    for idx, c in enumerate(valid_courses.course_id):
        course_id = int(c)
        course_num = idx
        course_second_level_cat = course_latest[course_latest.course_id == course_id][
            "second_level_category"
        ].iloc[0]
        is_valid_instructor = False
        idx_instr = 0
        while is_valid_instructor == False:
            course_instr = teach_latest.loc[teach_latest.course_id == course_id].iloc[
                idx_instr
            ]["instructor_id"]
            idx_instr += 1
            if course_instr in instructor_list:
                is_valid_instructor = True
        # print(str(course_instr) + "\n")
        courses_info[course_id] = {
            "num": course_num,
            "s_category": course_second_level_cat,
            "instructor": course_instr,
        }
        course_num_to_id[idx] = course_id
    return courses_info, course_num_to_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Config file.",
        default="config/coco/graph_reasoning/preprocess.json",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = edict(json.load(f))

    os.makedirs(config.processed_data_dir, exist_ok=True)

    course_latest = read_courses(config)
    teach_latest = read_teach(config, course_latest)
    instructor_ids = get_instructors_ids(teach_latest)
    valid_courses = get_valid_courses(teach_latest, instructor_ids, course_latest)
    student_enrolments = get_student_enrolments(config, valid_courses)
    course_skills = get_course_skill(config)

    instructor_id_to_num = write_entity(config, instructor_ids, "instructors")
    course_id_to_num = write_entity(config, valid_courses.course_id, "courses")

    courses_info, course_num_to_id = get_course_info(
        valid_courses,
        course_latest,
        teach_latest,
        instructor_ids,
    )

    categories = (
        course_latest.groupby(["first_level_category", "second_level_category"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
    )

    valid_categories = categories.loc[categories.second_level_category != "other"]
    s_level_categories = valid_categories.second_level_category.unique()
    f_level_categories = valid_categories.first_level_category.unique()

    f_level_category_to_num = {}
    num_to_f_level_category = {}
    for i, l in enumerate(f_level_categories):
        f_level_category_to_num[l] = i
        num_to_f_level_category[i] = l

    s_level_category_to_num = {}
    num_to_s_level_category = {}
    for i, s in enumerate(s_level_categories):
        s_level_category_to_num[s] = i
        num_to_s_level_category[i] = s

    with open(
        os.path.join(config.processed_data_dir, "second_categories.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        for i in range(len(s_level_categories)):
            f.write(num_to_s_level_category[i] + "\n")

    with open(
        os.path.join(config.processed_data_dir, "first_categories.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        for i in range(len(f_level_categories)):
            f.write(num_to_f_level_category[i] + "\n")

    student_id_to_num = {}
    num_to_student_id = {}
    for i, learner_id in enumerate(student_enrolments.learner_id.unique()):
        student_id_to_num[learner_id] = i
        num_to_student_id[i] = learner_id
    with open(
        os.path.join(config.processed_data_dir, "learners.txt"), "w", encoding="utf-8"
    ) as f:
        for i in range(len(student_id_to_num)):
            f.write(str(num_to_student_id[i]) + "\n")
    files = [
        "course_instructor.txt",
        "course_scategory.txt",
        "scategory_fcategory.txt",
        "enrolments.txt",
    ]
    for f in files:
        f = os.path.join(config.processed_data_dir, f)
        if os.path.exists(f):
            os.remove(f)

    for i in range(len(valid_courses.course_id)):
        course_id = course_num_to_id[i]
        course_info = courses_info[course_id]
        with open(
            os.path.join(config.processed_data_dir, "course_instructor.txt"),
            "a",
            encoding="utf-8",
        ) as f:
            f.write(str(instructor_id_to_num[course_info["instructor"]]) + "\n")
        with open(
            os.path.join(config.processed_data_dir, "course_scategory.txt"),
            "a",
            encoding="utf-8",
        ) as f:
            f.write(str(s_level_category_to_num[course_info["s_category"]]) + "\n")

    for i in range(len(s_level_categories)):
        s_cat = num_to_s_level_category[i]
        f_cat = valid_categories.loc[
            valid_categories.second_level_category == s_cat
        ].iloc[0]["first_level_category"]
        f_cat_num = f_level_category_to_num[f_cat]
        with open(
            os.path.join(config.processed_data_dir, "scategory_fcategory.txt"),
            "a",
            encoding="utf-8",
        ) as f:
            f.write(str(f_cat_num) + "\n")

    enrol_dict = {}
    with open(
        os.path.join(config.processed_data_dir, "enrolments.txt"), "a", encoding="utf-8"
    ) as f:
        for idx in student_enrolments.index:
            learner_id = student_enrolments.learner_id[idx]
            learner_id = student_id_to_num[learner_id]
            if learner_id not in enrol_dict:
                enrol_dict[learner_id] = []
            course_id = int(student_enrolments.course_id[idx])
            course_id = course_id_to_num[course_id]
            enrol_dict[learner_id].append(course_id)
            f.write(f"{learner_id} {course_id}\n")

    files = ["course_skills.txt", "skills.txt"]
    for f in files:
        f = os.path.join(config.processed_data_dir, f)
        if os.path.exists(f):
            os.remove(f)

    course_skills_dict = defaultdict(list)
    course_skills_freq = Counter()
    max_course_skills_count = config.max_concept_prop * len(course_id_to_num)
    skills = set()
    skill_id_to_num = {}
    for idx in course_skills.index:
        course_id = course_skills.course_id[idx]
        skill = course_skills.skills[idx]
        course_skills_dict[course_id].append(skill)
        course_skills_freq[skill] += 1
        skills.add(skill)

    # Define the valid skills as the ones that are in less than max_course_concept_count courses
    valid_skills = set(
        [
            s
            for s in course_skills_freq
            if course_skills_freq[s] <= max_course_skills_count
        ]
    )

    with open(os.path.join(config.processed_data_dir, "skills.txt"), "a") as f:
        for i, s in enumerate(skills):
            skill_id_to_num[s] = i
            f.write(f"{s}\n")

    for i in range(len(valid_courses.course_id)):
        course_id = course_num_to_id[i]
        course_skills = course_skills_dict[course_id]
        skill_nums = []
        for cs in course_skills:
            # if the skill is valid, add it to the list of skills
            if cs in valid_skills:
                skill_nums.append(str(skill_id_to_num[cs]))

        with open(
            os.path.join(config.processed_data_dir, "course_skills.txt"),
            "a",
            encoding="utf-8",
        ) as f:
            f.write(" ".join(skill_nums) + "\n")

    learner_skills = defaultdict(Counter)
    learner_skills_freq = Counter()
    max_learner_skills_count = config.max_concept_prop * len(student_id_to_num)
    # Iterate over the rows of student_enrolments and fill the learner_skills dictionary
    for idx in student_enrolments.index:
        learner_id = student_enrolments.learner_id[idx]
        learner_id = student_id_to_num[learner_id]
        course_id = student_enrolments.course_id[idx]
        # course_id = course_id_to_num[course_id]
        skills = course_skills_dict[course_id]
        for s in skills:
            if s in valid_skills:
                s = skill_id_to_num[s]
                learner_skills[learner_id][s] += 1
                learner_skills_freq[s] += 1

    # rank the skills by frequency
    for learner in learner_skills:
        learner_skills[learner] = learner_skills[learner].most_common()

    # Define the valid skills as the ones that are in less than max_learner_skills_count courses
    valid_skills = set(
        [
            s
            for s in learner_skills_freq
            if learner_skills_freq[s] <= max_learner_skills_count
        ]
    )

    with open(os.path.join(config.processed_data_dir, "learner_skills.txt"), "w") as f:
        for i in range(len(student_id_to_num)):
            # write the categories that the user is interested in, ignore the frequency
            skills_list = [
                str(skill) for skill, _ in learner_skills[i] if skill in valid_skills
            ]
            f.write(" ".join(skills_list) + "\n")


if __name__ == "__main__":
    main()
