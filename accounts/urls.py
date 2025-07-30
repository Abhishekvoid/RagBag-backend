from django.urls import path
from .views import RegisterAPIView, ChatSessionRetriveView, ChatMessageView, ChatSessionView,  SubjectListCreateView,  SubjectDetailView, ChapterDetailView, ChapterListCreateView, DocumentListCreateView, DocumentDetailView


urlpatterns = [
    path('register/', RegisterAPIView.as_view(), name='custom-register'),
    path('chatsessions/',ChatSessionView.as_view(), name='chatsessions-list-create' ),
    path('chatsessions/<uuid:id>/',ChatSessionRetriveView.as_view(), name='chatsession-detail'),
    path('chatmessage/', ChatMessageView.as_view(), name='chatmessage'),


    # Subject endpoints
    path('subjects/', SubjectListCreateView.as_view(), name='subjects-list-create'),
    path('subjects/<uuid:id>/', SubjectDetailView.as_view(), name='subject-detail'),

    # Chapter endpoints
    path('chapters/', ChapterListCreateView.as_view(), name='chapters-list-create'),
    path('chapters/<uuid:id>/', ChapterDetailView.as_view(), name='chapter-detail'),

    # Document endpoints
    path('documents/', DocumentListCreateView.as_view(), name='documents-list-create'),
    path('documents/<uuid:id>/', DocumentDetailView.as_view(), name='document-detail'),
]

