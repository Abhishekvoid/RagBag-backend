from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [('accounts', '0013_documentpage_s3_object_key')]
    operations = [
        migrations.AddField('document', 'coverage_audit', models.JSONField(default=dict, blank=True)),
        migrations.AddField('document', 'active_coverage', models.JSONField(default=dict, blank=True)),
        migrations.AddField('documentindexversion', 'coverage_audit', models.JSONField(default=dict, blank=True)),
        migrations.AddField('documentpage', 'needs_review', models.BooleanField(default=False)),
        migrations.AddField('documentpage', 'review_reason', models.CharField(max_length=100, blank=True)),
        migrations.AlterField('documentpage', 'text_source', models.CharField(max_length=10, default='layer',
            choices=[('layer', 'Text layer'), ('vision', 'Vision'), ('fallback', 'Fallback'), ('ocr', 'Local OCR')])),
    ]
