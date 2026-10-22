#!/usr/bin/env python3
"""Add the formal Place Project to one explicitly selected teaching course."""
import argparse
import json
from pathlib import Path
from sqlalchemy import text
from apps.api.database import create_postgres_engine, create_session_factory
from robotics.projects import PostgreSQLTeachingProjectRepository, TeachingProjectService, TrainingProfile
from scripts.always_on_runtime import load_environment, CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--course-code', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    environment = load_environment(CONFIG)
    engine = create_postgres_engine(database_url=environment['DATABASE_URL'])
    service = TeachingProjectService(PostgreSQLTeachingProjectRepository(create_session_factory(engine)))
    with engine.connect() as connection:
        courses = list(connection.execute(text('SELECT course_id,owner_instructor_user_id FROM courses WHERE code=:code AND enabled=true'), {'code': args.course_code}).mappings())
        if len(courses) != 1:
            raise ValueError('Select exactly one enabled course')
        course = courses[0]
        classes = list(connection.execute(text('SELECT class_id FROM classes WHERE course_id=:course AND enabled=true'), {'course': course['course_id']}).scalars())
    if not classes:
        raise ValueError('The course has no enabled teaching class')
    actor = course['owner_instructor_user_id']
    matches = [item for item in service.list_projects(course['course_id'], actor) if item.project.environment_type == 's1_place']
    if matches:
        print(json.dumps({'result': 'existing', 'projects': [{'project_id': item.project.project_id, 'enabled': item.project.enabled, 'config_version': item.project.config_version} for item in matches]}))
        return
    preview = {'course_code': args.course_code, 'course_id': course['course_id'], 'class_ids': classes, 'title': '搬运放置', 'environment_type': 's1_place', 'profile': 'standard', 'video_enabled': True}
    if not args.apply:
        print(json.dumps({'result': 'preview', **preview}, ensure_ascii=False))
        return
    access = service.create_project(course['course_id'], actor, title=preview['title'], description='Formal S1 Place transport-and-hold training from a validated lifted grasp.', instructions='Select a saved reward revision, pass preflight, and inspect actual Run evidence. Full grasp, release and retreat are outside this task.', starter_code='', training_profile=TrainingProfile.STANDARD, video_enabled=True, enabled=True, environment_type='s1_place')
    record = {'result': 'created', **preview, 'project_id': access.project.project_id, 'config_version': access.project.config_version, 'rollback': 'Disable only this new Project through the canonical Project service; preserve existing Runs.'}
    output = Path('var/validation/20261010-place-project.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
