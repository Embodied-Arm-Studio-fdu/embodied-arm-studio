#!/usr/bin/env python3
"""Safely preview or converge one namespaced midterm teaching demo."""

import argparse
from dataclasses import dataclass
import getpass
import re
import sys
from typing import Callable, Dict, List, Optional, TextIO

from apps.api.database import (
    DatabaseConfigurationError,
    create_postgres_engine,
    create_session_factory,
)
from robotics.auth import (
    AuthenticationService,
    DuplicateUsername,
    PasswordHasher,
    PostgreSQLUserRepository,
    User,
)
from robotics.courses import (
    CourseService,
    PostgreSQLCourseRepository,
    TeachingConflict,
)
from robotics.organizations import (
    MembershipNotFound,
    OrganizationConflict,
    OrganizationRole,
    OrganizationService,
    PostgreSQLOrganizationRepository,
)
from robotics.projects import (
    PostgreSQLTeachingProjectRepository,
    TeachingProjectConflict,
    TeachingProjectService,
    TrainingProfile,
    current_config_version,
    evaluation_rubric,
    parameter_schema,
)


_NAMESPACE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class BootstrapConflict(RuntimeError):
    """A namespaced business key exists with different immutable facts."""


@dataclass(frozen=True)
class DemoNames:
    namespace: str
    administrator_username: str
    instructor_username: str
    student_username: str
    organization_slug: str
    organization_name: str
    course_code: str
    course_title: str
    class_code: str
    class_name: str
    class_term: str
    short_project_title: str
    standard_project_title: str
    reach_project_title: str
    grasp_project_title: str
    lift_project_title: str
    lower_release_project_title: str


@dataclass(frozen=True)
class BootstrapSummary:
    names: DemoNames
    project_facts: List[Dict[str, object]]


def demo_names(namespace: str) -> DemoNames:
    value = namespace.strip().lower() if isinstance(namespace, str) else ""
    if len(value) > 48 or _NAMESPACE.fullmatch(value) is None:
        raise ValueError(
            "namespace must contain 1-48 lowercase letters, digits, and single hyphens"
        )
    label = value.upper()
    return DemoNames(
        namespace=value,
        administrator_username="%s-administrator" % value,
        instructor_username="%s-instructor" % value,
        student_username="%s-student" % value,
        organization_slug=value,
        organization_name="Embodied Arm Midterm Demo [%s]" % value,
        course_code="S1-PUSH-%s" % label,
        course_title="S1 Push Midterm Demo [%s]" % value,
        class_code="MIDTERM-%s" % label,
        class_name="Midterm [%s]" % value,
        class_term="midterm-%s" % value,
        short_project_title="入门训练与演示",
        standard_project_title="推动训练",
        reach_project_title="触达训练",
        grasp_project_title="抓取训练",
        lift_project_title="抓取抬起",
        lower_release_project_title="抓取放下",
    )


class MidtermDemoBootstrap:
    """Converge additive demo facts through canonical application services."""

    def __init__(
        self,
        session_factory,
        namespace: str,
        *,
        password_reader: Callable[[str], str] = getpass.getpass,
        output: TextIO = sys.stdout,
    ) -> None:
        self.names = demo_names(namespace)
        self.password_reader = password_reader
        self.output = output
        self.users = PostgreSQLUserRepository(session_factory)
        self.authentication = AuthenticationService(self.users, PasswordHasher())
        self.organizations = OrganizationService(
            PostgreSQLOrganizationRepository(session_factory)
        )
        self.courses = CourseService(PostgreSQLCourseRepository(session_factory))
        self.projects = TeachingProjectService(
            PostgreSQLTeachingProjectRepository(session_factory)
        )
        self.actions: List[str] = []

    def run(self, *, apply: bool) -> BootstrapSummary:
        expected_users = self._expected_users()
        existing_users: Dict[str, Optional[User]] = {}
        for username, display_name in expected_users.items():
            user = self.users.find_by_username(username)
            if user is not None:
                self._require_user(user, display_name)
            existing_users[username] = user

        if not apply:
            self._preview(existing_users)
            return self._public_summary([], planned=True)

        passwords = self._read_new_passwords(existing_users)
        resolved_users: Dict[str, User] = {}
        for username, display_name in expected_users.items():
            current = existing_users[username]
            if current is None:
                try:
                    current = self.authentication.create_user(
                        username, display_name, passwords[username]
                    )
                except DuplicateUsername as exc:
                    raise BootstrapConflict(
                        "username %s was created concurrently; rerun to verify facts"
                        % username
                    ) from exc
                self._record("created user %s" % username)
            else:
                self._record("reused user %s" % username)
            resolved_users[username] = current

        administrator = resolved_users[self.names.administrator_username]
        instructor = resolved_users[self.names.instructor_username]
        student = resolved_users[self.names.student_username]
        organization = self._ensure_organization(administrator)
        self._ensure_membership(
            organization.organization_id,
            administrator,
            OrganizationRole.ADMINISTRATOR,
        )
        self._ensure_membership(
            organization.organization_id, instructor, OrganizationRole.INSTRUCTOR
        )
        self._ensure_membership(
            organization.organization_id, student, OrganizationRole.STUDENT
        )
        course = self._ensure_course(
            organization.organization_id, administrator.user_id, instructor.user_id
        )
        teaching_class = self._ensure_class(
            course.course_id, administrator.user_id, instructor.user_id
        )
        self._ensure_enrollment(
            teaching_class.class_id, student.user_id, administrator.user_id
        )
        projects = [
            self._ensure_project(
                course.course_id,
                administrator.user_id,
                self.names.short_project_title,
                TrainingProfile.SHORT,
                False,
            ),
            self._ensure_project(
                course.course_id,
                administrator.user_id,
                self.names.standard_project_title,
                TrainingProfile.STANDARD,
                True,
            ),
            self._ensure_project(
                course.course_id,
                administrator.user_id,
                self.names.reach_project_title,
                TrainingProfile.STANDARD,
                True,
                "s1_reach",
            ),
            self._ensure_project(
                course.course_id, administrator.user_id,
                self.names.grasp_project_title, TrainingProfile.STANDARD, True,
                "s1_grasp",
            ),
            self._ensure_project(
                course.course_id, administrator.user_id,
                self.names.lift_project_title, TrainingProfile.STANDARD, True,
                "s1_lift",
            ),
            self._ensure_project(
                course.course_id, administrator.user_id,
                self.names.lower_release_project_title, TrainingProfile.STANDARD, True,
                "s1_lower_release",
            ),
        ]
        return self._public_summary(projects)

    def _expected_users(self) -> Dict[str, str]:
        value = self.names.namespace
        return {
            self.names.administrator_username: "Midterm Demo Administrator [%s]" % value,
            self.names.instructor_username: "Midterm Demo Instructor [%s]" % value,
            self.names.student_username: "Midterm Demo Student [%s]" % value,
        }

    @staticmethod
    def _require_user(user: User, display_name: str) -> None:
        if user.display_name != display_name or not user.enabled:
            raise BootstrapConflict(
                "username %s exists with different public facts" % user.username
            )

    def _read_new_passwords(
        self, existing: Dict[str, Optional[User]]
    ) -> Dict[str, str]:
        values: Dict[str, str] = {}
        for username, user in existing.items():
            if user is not None:
                continue
            first = self.password_reader("Password for new user %s: " % username)
            second = self.password_reader("Confirm password for %s: " % username)
            if not first:
                raise ValueError("password must not be empty")
            if first != second:
                raise ValueError("password confirmation does not match")
            values[username] = first
        return values

    def _preview(self, existing: Dict[str, Optional[User]]) -> None:
        self._write("Mode: dry-run (zero writes)")
        for username, user in existing.items():
            self._write(
                "%s user %s" % ("reuse" if user is not None else "create", username)
            )
        organization = self.organizations.repository.find_by_slug(
            self.names.organization_slug
        )
        if organization is not None:
            self._require_organization(organization)
            self._write("reuse organization %s" % self.names.organization_slug)
        else:
            self._write("create organization %s" % self.names.organization_slug)
        self._write("converge three enabled memberships")
        self._write("converge course %s" % self.names.course_code)
        self._write("converge class %s and enabled Student enrollment" % self.names.class_code)
        self._write("converge project %s (short, video=false)" % self.names.short_project_title)
        self._write("converge project %s (standard, video=true)" % self.names.standard_project_title)
        self._write("converge project %s (reach standard, video=true)" % self.names.reach_project_title)
        self._write("converge project %s (grasp standard, video=true)" % self.names.grasp_project_title)
        self._write("converge project %s (lift standard, video=true)" % self.names.lift_project_title)
        self._write("converge project %s (lower/release standard, video=true)" % self.names.lower_release_project_title)

    def _ensure_organization(self, administrator: User):
        existing = self.organizations.repository.find_by_slug(
            self.names.organization_slug
        )
        if existing is None:
            try:
                organization, _ = self.organizations.create_organization(
                    self.names.organization_slug,
                    self.names.organization_name,
                    administrator.user_id,
                )
            except OrganizationConflict as exc:
                raise BootstrapConflict(
                    "organization slug conflicted; rerun to verify facts"
                ) from exc
            self._record("created organization %s" % organization.slug)
            return organization
        self._require_organization(existing)
        self._record("reused organization %s" % existing.slug)
        return existing

    def _require_organization(self, organization) -> None:
        if organization.name != self.names.organization_name or not organization.enabled:
            raise BootstrapConflict(
                "organization slug %s exists with different facts"
                % self.names.organization_slug
            )

    def _ensure_membership(self, organization_id: str, user: User, role) -> None:
        try:
            existing = self.organizations.get_membership(
                organization_id, user.user_id
            )
        except MembershipNotFound:
            try:
                self.organizations.add_membership(
                    organization_id, user.user_id, role
                )
            except OrganizationConflict as exc:
                raise BootstrapConflict(
                    "membership for %s conflicted; rerun to verify facts"
                    % user.username
                ) from exc
            self._record("created %s membership for %s" % (role.value, user.username))
            return
        if existing.role is not role or not existing.enabled:
            raise BootstrapConflict(
                "membership for %s exists with different facts" % user.username
            )
        self._record("reused %s membership for %s" % (role.value, user.username))

    def _ensure_course(
        self, organization_id: str, actor_user_id: str, instructor_user_id: str
    ):
        matches = [
            item
            for item in self.courses.list_courses(organization_id, actor_user_id)
            if item.code == self.names.course_code
        ]
        if len(matches) > 1:
            raise BootstrapConflict("course code is ambiguous")
        if not matches:
            try:
                course = self.courses.create_course(
                    organization_id,
                    self.names.course_code,
                    self.names.course_title,
                    "S1 Push midterm demonstration course.",
                    instructor_user_id,
                    actor_user_id,
                )
            except TeachingConflict as exc:
                raise BootstrapConflict(
                    "course code exists with different or concurrent facts"
                ) from exc
            self._record("created course %s" % course.code)
            return course
        course = matches[0]
        if (
            course.organization_id != organization_id
            or course.title != self.names.course_title
            or course.description != "S1 Push midterm demonstration course."
            or course.owner_instructor_user_id != instructor_user_id
            or not course.enabled
        ):
            raise BootstrapConflict("course code exists with different facts")
        self._record("reused course %s" % course.code)
        return course

    def _ensure_class(
        self, course_id: str, actor_user_id: str, instructor_user_id: str
    ):
        matches = [
            item
            for item in self.courses.list_classes(course_id, actor_user_id)
            if item.code == self.names.class_code and item.term == self.names.class_term
        ]
        if len(matches) > 1:
            raise BootstrapConflict("class business key is ambiguous")
        if not matches:
            try:
                teaching_class = self.courses.create_class(
                    course_id,
                    self.names.class_code,
                    self.names.class_name,
                    self.names.class_term,
                    instructor_user_id,
                    actor_user_id,
                )
            except TeachingConflict as exc:
                raise BootstrapConflict(
                    "class business key exists with different or concurrent facts"
                ) from exc
            self._record("created class %s" % teaching_class.code)
            return teaching_class
        teaching_class = matches[0]
        if (
            teaching_class.course_id != course_id
            or teaching_class.name != self.names.class_name
            or teaching_class.instructor_user_id != instructor_user_id
            or not teaching_class.enabled
        ):
            raise BootstrapConflict("class business key exists with different facts")
        self._record("reused class %s" % teaching_class.code)
        return teaching_class

    def _ensure_enrollment(
        self, class_id: str, student_user_id: str, actor_user_id: str
    ) -> None:
        matches = [
            item
            for item in self.courses.list_enrollments(class_id, actor_user_id)
            if item.user_id == student_user_id
        ]
        if not matches:
            try:
                self.courses.put_enrollment(
                    class_id, student_user_id, True, actor_user_id
                )
            except TeachingConflict as exc:
                raise BootstrapConflict(
                    "Student enrollment conflicted; rerun to verify facts"
                ) from exc
            self._record("created enabled Student enrollment")
            return
        if len(matches) != 1 or not matches[0].enabled:
            raise BootstrapConflict("Student enrollment exists with different facts")
        self._record("reused enabled Student enrollment")

    def _ensure_project(
        self,
        course_id: str,
        actor_user_id: str,
        title: str,
        profile: TrainingProfile,
        video_enabled: bool,
        environment_type: str = "s1_push",
    ):
        matches = [
            item.project
            for item in self.projects.list_projects(course_id, actor_user_id)
            if item.project.title == title
        ]
        if len(matches) > 1:
            raise BootstrapConflict("teaching Project title is ambiguous")
        task_labels = {
            "s1_push": "S1 Push", "s1_reach": "S1 Reach",
            "s1_grasp": "S1 Grasp", "s1_lift": "S1 Grasp & Lift",
            "s1_lower_release": "S1 Lower & Release",
        }
        task_label = task_labels[environment_type]
        description = "Canonical %s %s training profile." % (task_label, profile.value)
        instructions = "Set only seed and inspect real training evidence."
        desired_version = current_config_version(profile)
        if not matches:
            try:
                access = self.projects.create_project(
                    course_id,
                    actor_user_id,
                    title=title,
                    description=description,
                    instructions=instructions,
                    starter_code="",
                    training_profile=profile,
                    video_enabled=video_enabled,
                    enabled=True,
                    environment_type=environment_type,
                )
            except TeachingProjectConflict as exc:
                raise BootstrapConflict(
                    "teaching Project title exists with different or concurrent facts"
                ) from exc
            self._record("created %s Project %s" % (profile.value, title))
            return access.project
        project = matches[0]
        if (
            profile is TrainingProfile.STANDARD
            and project.config_version < desired_version
            and project.description == description
            and project.instructions == instructions
            and project.starter_code == ""
            and project.training_profile is profile
            and self._schema_matches_environment(
                project.parameter_schema, environment_type, allow_legacy=True
            )
            and project.evaluation_rubric
            == evaluation_rubric(profile, project.config_version, environment_type)
            and project.enabled
            and project.created_by_user_id == actor_user_id
        ):
            while project.config_version < desired_version:
                changes = {"training_profile": TrainingProfile.STANDARD.value}
                if project.video_enabled is not video_enabled:
                    changes["video_enabled"] = video_enabled
                project = self.projects.update_project(
                    course_id,
                    project.project_id,
                    actor_user_id,
                    project.config_version,
                    changes,
                ).project
                self._record(
                    "upgraded standard Project to config version %d"
                    % project.config_version
                )
        if (
            project.description != description
            or project.instructions != instructions
            or project.starter_code != ""
            or project.training_profile is not profile
            or project.video_enabled is not video_enabled
            or not self._schema_matches_environment(
                project.parameter_schema, environment_type, allow_legacy=True
            )
            or project.evaluation_rubric
            != evaluation_rubric(profile, desired_version, environment_type)
            or project.environment_type != environment_type
            or project.config_version != desired_version
            or not project.enabled
            or project.created_by_user_id != actor_user_id
        ):
            raise BootstrapConflict("teaching Project title exists with different facts")
        self._record("reused %s Project %s" % (profile.value, title))
        return project

    @staticmethod
    def _schema_matches_environment(schema, environment_type, *, allow_legacy=False):
        expected = parameter_schema(environment_type)
        if schema == expected:
            return True
        if allow_legacy and environment_type == "s1_push":
            legacy = dict(expected)
            legacy.pop("x-environment-type", None)
            return schema == legacy
        return False

    def _record(self, message: str) -> None:
        self.actions.append(message)
        self._write(message)

    def _write(self, message: str) -> None:
        print(message, file=self.output)

    def _public_summary(self, projects, *, planned: bool = False) -> BootstrapSummary:
        if planned:
            facts = [
                {
                    "title": self.names.short_project_title,
                    "profile": TrainingProfile.SHORT.value,
                    "version": 1,
                },
                {
                    "title": self.names.standard_project_title,
                    "profile": TrainingProfile.STANDARD.value,
                    "version": current_config_version(TrainingProfile.STANDARD),
                },
                {
                    "title": self.names.reach_project_title,
                    "profile": TrainingProfile.STANDARD.value,
                    "version": current_config_version(TrainingProfile.STANDARD),
                },
                {
                    "title": self.names.grasp_project_title,
                    "profile": TrainingProfile.STANDARD.value,
                    "version": current_config_version(TrainingProfile.STANDARD),
                },
                {
                    "title": self.names.lift_project_title,
                    "profile": TrainingProfile.STANDARD.value,
                    "version": current_config_version(TrainingProfile.STANDARD),
                },
                {
                    "title": self.names.lower_release_project_title,
                    "profile": TrainingProfile.STANDARD.value,
                    "version": current_config_version(TrainingProfile.STANDARD),
                },
            ]
        else:
            facts = [
                {
                    "title": project.title,
                    "profile": project.training_profile.value,
                    "version": project.config_version,
                }
                for project in projects
            ]
        self._write("Public summary:")
        self._write(
            "users=%s,%s,%s"
            % (
                self.names.administrator_username,
                self.names.instructor_username,
                self.names.student_username,
            )
        )
        self._write("organization_slug=%s" % self.names.organization_slug)
        self._write("course_code=%s" % self.names.course_code)
        self._write("class_code=%s" % self.names.class_code)
        for item in facts:
            self._write(
                "project=%s profile=%s version=%s"
                % (item["title"], item["profile"], item["version"])
            )
        return BootstrapSummary(self.names, facts)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview or safely converge namespaced midterm demo data"
    )
    parser.add_argument("--namespace", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="preview only (default)")
    mode.add_argument("--apply", action="store_true", help="perform additive writes")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    engine = None
    try:
        engine = create_postgres_engine()
        bootstrap = MidtermDemoBootstrap(
            create_session_factory(engine), args.namespace
        )
        bootstrap.run(apply=bool(args.apply))
        return 0
    except BootstrapConflict as exc:
        print("Bootstrap conflict: %s" % exc, file=sys.stderr)
        return 3
    except (ValueError, DatabaseConfigurationError) as exc:
        print("Bootstrap request rejected: %s" % exc, file=sys.stderr)
        return 2
    except Exception:
        print("Bootstrap failed: PostgreSQL bootstrap dependency is unavailable", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
