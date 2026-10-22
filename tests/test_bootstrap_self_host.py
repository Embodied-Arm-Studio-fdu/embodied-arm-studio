from dataclasses import replace
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.bootstrap_self_host import SelfHostBootstrap
from scripts.bootstrap_midterm_demo import BootstrapConflict, MidtermDemoBootstrap, demo_names
from robotics.projects import TrainingProfile


class SelfHostBootstrapTest(unittest.TestCase):
    def bootstrap(self, existing):
        instance = SelfHostBootstrap.__new__(SelfHostBootstrap)
        instance.names = demo_names('studio')
        instance.output = io.StringIO()
        instance.users = Mock()
        instance.users.find_by_username.return_value = SimpleNamespace(user_id='admin')
        instance.organizations = Mock()
        instance.organizations.repository.find_by_slug.return_value = SimpleNamespace(organization_id='org')
        instance.courses = Mock()
        instance.courses.list_courses.return_value = [SimpleNamespace(code=instance.names.course_code, course_id='course')]
        instance.projects = Mock()
        instance.projects.list_projects.return_value = [SimpleNamespace(project=value) for value in existing]
        return instance

    def test_fresh_bootstrap_adds_place_using_canonical_project_service(self):
        instance = self.bootstrap([])
        with patch.object(MidtermDemoBootstrap, 'run', return_value='summary'):
            self.assertEqual('summary', instance.run(apply=True))
        kwargs = instance.projects.create_project.call_args[1]
        self.assertEqual('s1_place', kwargs['environment_type'])
        self.assertEqual(TrainingProfile.STANDARD, kwargs['training_profile'])
        self.assertTrue(kwargs['video_enabled'])

    def test_repeat_bootstrap_does_not_duplicate_or_rewrite_place(self):
        existing = SimpleNamespace(environment_type='s1_place', enabled=True, video_enabled=True, training_profile=TrainingProfile.STANDARD)
        instance = self.bootstrap([existing])
        with patch.object(MidtermDemoBootstrap, 'run'):
            instance.run(apply=True)
        instance.projects.create_project.assert_not_called()
        existing.enabled = False
        with patch.object(MidtermDemoBootstrap, 'run'), self.assertRaises(BootstrapConflict):
            instance.run(apply=True)

    def test_preview_does_not_create_the_additional_project(self):
        instance = self.bootstrap([])
        with patch.object(MidtermDemoBootstrap, 'run'):
            instance.run(apply=False)
        instance.projects.create_project.assert_not_called()
