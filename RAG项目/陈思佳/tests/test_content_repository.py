from src.edu_rag_ingest.storage.content_repository import Database, GeneratedQuestionSet, LessonPlan


def test_database_creates_and_persists_content(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'education.db'}")
    database.create_tables()

    with database.session() as session:
        lesson = LessonPlan(subject="语文", grade="九年级", chapter="阅读", title="阅读课", content="教学设计")
        question = GeneratedQuestionSet(
            subject="语文",
            grade="九年级",
            chapter="阅读",
            knowledge_point="人物形象",
            question_type="选择题",
            difficulty="中等",
            content="题目内容",
        )
        session.add_all([lesson, question])
        session.commit()

    with database.session() as session:
        assert session.query(LessonPlan).count() == 1
        assert session.query(GeneratedQuestionSet).count() == 1
