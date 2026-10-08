from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("accounts", "0010_documentpage")]

    operations = [
        migrations.AddField(
            model_name="chatmessage",
            name="is_unanswered",
            field=models.BooleanField(default=False),
        ),
    ]
