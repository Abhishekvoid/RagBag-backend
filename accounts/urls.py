from django.urls import path
from .views import RegisterAPIView, ChatSessionRetriveView, ChatMessageView, ChatSessionView


urlpatterns = [
    path('register/', RegisterAPIView.as_view(), name='custom-register'),
    path('chatsessions/',ChatSessionView.as_view(), name='chatsessions-list-create' ),
    path('chatsessions/<uuid:id>/',ChatSessionRetriveView.as_view(), name='chatsession-detail'),
    path('chatmessage/', ChatMessageView.as_view(), name='chatmessage'),
]
