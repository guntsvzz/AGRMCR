import os
import json
import argparse

from easydict import EasyDict as edict


def save_learners(datadir, savedir, name):
    """Save learners with recbole format

    Args:
        datadir (str): path of the processed dataset
        savedir (str): path of recbole dataset
        name (str): name of the dataset

    Returns:
        list: list of users
    """
    users = []
    with open(os.path.join(datadir, "learners.txt"), "r") as f:
        for line in f:
            users.append(line.strip())

    with open(os.path.join(savedir, name + ".user"), "w") as f:
        f.write("user_id:token\n")
        for user_id in users:
            f.write(f"L_{user_id}\n")
    return users


def save_courses(datadir, savedir, name):
    """Save courses with recbole format

    Args:
        datadir (str): path of the processed dataset
        savedir (str): path of recbole dataset
        name (str): name of the dataset

    Returns:
        list: list of courses
    """
    courses = []
    with open(os.path.join(datadir, "courses.txt"), "r") as f:
        for line in f:
            courses.append(line.strip())

    with open(os.path.join(savedir, name + ".item"), "w") as f:
        f.write("item_id:token\n")
        for item_id in courses:
            f.write(f"C_{item_id}\n")
    return courses


def save_course_entity(savedir, name, courses, interacted_courses):
    """Save coco course entity to to recbole format.

    Args:
        savedir (str): path of the recbole dataset
        name (str): name of the dataset
        courses (list): list of courses
        interacted_courses (set): set of courses that have been interacted with
    """
    with open(os.path.join(savedir, name + ".link"), "w") as f:
        f.write("item_id:token\tentity_id:token\n")
        for course in courses:
            if course in interacted_courses:
                f.write(f"C_{course}\tC_{course}\n")


def read_course_instructors(datadir, kg_triplets, courses, interacted_courses):
    """Update kg triplets with instructors.

    Args:
        datadir (str): path of the processed dataset
        kg_triplets (list): list of kg triplets
        courses (list): list of courses
        interacted_courses (set): set of courses that have been interacted with
    """
    with open(os.path.join(datadir, "course_instructor.txt"), "r") as f:
        for i, line in enumerate(f):
            instructor = line.strip()
            course = courses[int(i)]
            if instructor and course in interacted_courses:
                kg_triplets.append(
                    [
                        "C_" + course,
                        "instructor",
                        "I_" + instructor,
                    ]
                )


def read_course_category(datadir, kg_triplets, courses, interacted_courses):
    """Update kg triplets to with category.

    Args:
        datadir (str): path of the processed dataset
        kg_triplets (list): list of kg triplets
        courses (list): list of courses
        interacted_courses (set): set of courses that have been interacted with
    """
    with open(os.path.join(datadir, "course_scategory.txt"), "r") as f:
        for i, line in enumerate(f):
            category = line.strip()
            course = courses[int(i)]
            if category and course in interacted_courses:
                kg_triplets.append(
                    [
                        "C_" + courses[int(i)],
                        "category",
                        "Ca_" + category,
                    ]
                )


def read_course_skills(datadir, kg_triplets, courses, interacted_courses):
    """Update kg triplets with skills.

    Args:
        datadir (str): path of the processed dataset
        kg_triplets (list): list of kg triplets
        courses (list): list of courses
        interacted_courses (set): set of courses that have been interacted with
    """
    with open(os.path.join(datadir, "course_skills.txt"), "r") as f:
        for i, line in enumerate(f):
            skills = line.strip()
            course = courses[int(i)]
            if skills and course in interacted_courses:
                for skill in skills.split():
                    kg_triplets.append(
                        [
                            "C_" + courses[int(i)],
                            "teaches_skill",
                            "S_" + skill,
                        ]
                    )


def read_learners_skills(datadir, kg_triplets, learners):
    """Update kg triplets with skills.

    Args:
        datadir (str): path of the processed dataset
        kg_triplets (list): list of kg triplets
        courses (list): list of courses
    """
    with open(os.path.join(datadir, "learner_skills.txt"), "r") as f:
        for i, line in enumerate(f):
            skills = line.strip()
            if skills:
                for skill in skills.split():
                    kg_triplets.append(
                        [
                            "L_" + learners[int(i)],
                            "has_skill",
                            "S_" + skill,
                        ]
                    )


def read_category_hierarchy(datadir, kg_triplets):
    """Update kg triplets to with category hierarchy.

    Args:
        datadir (str): path of the processed dataset
        kg_triplets (list): list of kg triplets
    """
    with open(os.path.join(datadir, "scategory_fcategory.txt"), "r") as f:
        for i, line in enumerate(f):
            pcategory = line.strip()
            if pcategory:
                kg_triplets.append(
                    [
                        "Ca_" + str(i),
                        "child_category",
                        "PCa_" + pcategory,
                    ]
                )


def save_kg_triplets(kg_triplets, savedir, name):
    """Save kg_triplets to file.

    Args:
        kg_triplets (list): list of triplets as a tuple (head_id, relation_id, tail_id)
        savedir (str): path of the recbole dataset
        name (str): name of the dataset
    """
    with open(os.path.join(savedir, name + ".kg"), "w") as f:
        f.write("head_id:token\trelation_id:token\ttail_id:token\n")
        for head_id, relation_id, tail_id in kg_triplets:
            f.write(f"{head_id}\t{relation_id}\t{tail_id}\n")


def save_enrolment(
    datadir, savedir, name, subset, learners, courses, interacted_courses
):
    """Save coco enrolments to file.

    Args:
        datadir (str): path of the processed dataset
        savedir (str): path of the recbole dataset
        name (str): name of the dataset
        subset (str): name of the subset
        learners (list): list of learners
        courses (list): list of courses
        interacted_courses (set): set of courses that have been interacted with
    """
    enrolments = []
    with open(os.path.join(datadir, subset + ".txt"), "r") as f:
        for line in f:
            enrolments.append([int(x) for x in line.split()])

    with open(os.path.join(savedir, name + "." + subset + ".inter"), "w") as f:
        f.write("user_id:token\titem_id:token\trating:float\n")
        for learner, course in enrolments:
            interacted_courses.add(courses[course])
            f.write(f"L_{learners[learner]}\tC_{courses[course]}\t1\n")


def format_pgpr_coco(datadir, savedir, dataset_name):
    """Format PGPR dataset to recbole format

    Args:
        datadir (str): path of the processed dataset
        savedir (str): path of the recbole dataset
        dataset_name (str): name of the dataset
    """

    learners = save_learners(datadir, savedir, dataset_name)
    courses = save_courses(datadir, savedir, dataset_name)

    subsets = ["train", "validation", "test"]

    # keep track of the items that have been interacted with, recbole does not support items that have not been interacted with but that are in the kg file
    interacted_courses = set()

    for subset in subsets:
        save_enrolment(
            datadir,
            savedir,
            dataset_name,
            subset,
            learners,
            courses,
            interacted_courses,
        )

    save_course_entity(savedir, dataset_name, courses, interacted_courses)
    kg_triplets = []
    read_course_instructors(datadir, kg_triplets, courses, interacted_courses)
    read_course_category(datadir, kg_triplets, courses, interacted_courses)
    read_course_skills(datadir, kg_triplets, courses, interacted_courses)
    read_learners_skills(datadir, kg_triplets, learners)
    read_category_hierarchy(datadir, kg_triplets)
    save_kg_triplets(kg_triplets, datadir, dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Config file.",
        default="config/coco/baselines/format.json",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = edict(json.load(f))

    # Creates the folder savedir if it does not exist
    os.makedirs(config.savedir, exist_ok=True)

    format_pgpr_coco(config.datadir, config.savedir, config.dataset_name)
