#!/usr/bin/env python3
"""Create isolated self-hosted accounts and all canonical S1 teaching projects."""
import argparse
from dataclasses import replace
import sys

from apps.api.database import create_postgres_engine, create_session_factory
from robotics.projects import TrainingProfile
from scripts.bootstrap_midterm_demo import MidtermDemoBootstrap, BootstrapConflict


class SelfHostBootstrap(MidtermDemoBootstrap):
    def __init__(self, session_factory, namespace, **kwargs):
        super().__init__(session_factory, namespace, **kwargs)
        self.names = replace(self.names,
            organization_name='Embodied Arm Studio [%s]' % namespace,
            course_title='S1 Simulation Lab [%s]' % namespace,
            class_name='Learning group [%s]' % namespace,
        )

    def _expected_users(self):
        return {self.names.administrator_username: 'Studio administrator',
                self.names.instructor_username: 'Studio instructor',
                self.names.student_username: 'Studio learner'}

    def run(self, *, apply):
        summary = super().run(apply=apply)
        if not apply:
            self._write('create or reuse S1 Place transport-and-hold Project')
            return summary
        administrator = self.users.find_by_username(self.names.administrator_username)
        organization = self.organizations.repository.find_by_slug(self.names.organization_slug)
        courses = [course for course in self.courses.list_courses(organization.organization_id, administrator.user_id) if course.code == self.names.course_code]
        if len(courses) != 1:
            raise BootstrapConflict('The self-hosted course could not be resolved uniquely')
        existing = [item.project for item in self.projects.list_projects(courses[0].course_id, administrator.user_id) if item.project.environment_type == 's1_place']
        if existing:
            if len(existing) != 1 or not existing[0].enabled or not existing[0].video_enabled or existing[0].training_profile != TrainingProfile.STANDARD:
                raise BootstrapConflict('Existing Place Project has incompatible settings; inspect it explicitly')
            self._write('reused S1 Place Project')
        else:
            self.projects.create_project(courses[0].course_id, administrator.user_id,
                title='Transport and place', description='Transport from a validated lifted grasp and hold at the target.',
                instructions='Save and check the reward draft before training; inspect formal evaluation evidence.',
                starter_code='', training_profile=TrainingProfile.STANDARD, video_enabled=True, enabled=True, environment_type='s1_place')
            self._write('created S1 Place Project')
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', default='studio')
    parser.add_argument('--apply', action='store_true', help='Create missing facts; otherwise preview only')
    args = parser.parse_args(argv)
    engine = None
    try:
        engine = create_postgres_engine()
        SelfHostBootstrap(create_session_factory(engine), args.namespace).run(apply=args.apply)
        return 0
    except Exception as exc:
        # Database errors may contain connection credentials; never print them.
        print('Bootstrap failed (%s). Check database readiness and existing namespace facts.' % type(exc).__name__, file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == '__main__':
    raise SystemExit(main())
