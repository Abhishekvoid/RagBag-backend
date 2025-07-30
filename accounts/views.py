from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, generics, permissions
from django.contrib.auth import get_user_model
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import RegisterSerializers, ChatMessageSerializer, ChatSessionSerializer, DocumentSerializer, SubjectSerializer, ChapterSerializer
import logging
from django.core.exceptions import ValidationError
from rest_framework.throttling import UserRateThrottle
from rest_framework.permissions import IsAuthenticated
from .models import ChatMessage, ChatSession, Document, Subject, Chapter

logger = logging.getLogger(__name__)
class RegisterAPIView(APIView):
    throttle_classes = [UserRateThrottle]  # Rate limiting
    
    def post(self, request, *args, **kwargs):
        try:
            serializer = RegisterSerializers(data=request.data)
            
            if not serializer.is_valid():
                logger.warning(f"Registration failed: {serializer.errors}")
                return Response({
                    'error': 'Invalid data provided',
                    'details': serializer.errors
                }, status=status.HTTP_400_BAD_REQUEST)
            
            user = serializer.save()
            refresh = RefreshToken.for_user(user)
            
            logger.info(f"User registered successfully: {user.email}")
            
            return Response({
                'user': {
                    'id': user.id,
                    'email': user.email,
                    'date_joined': user.date_joined.isoformat(),
                },
                'tokens': {
                    'access': str(refresh.access_token),
                    'refresh': str(refresh),
                }
            }, status=status.HTTP_201_CREATED)
            
        except ValidationError as e:
            logger.error(f"Validation error during registration: {e}")
            return Response({
                'error': 'Registration failed',
                'message': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)
            
        except Exception as e:
            logger.error(f"Unexpected error during registration: {e}")
            return Response({
                'error': 'Internal server error'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

# ------------ subject --------------


class SubjectListCreateView(generics.ListCreateAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = SubjectSerializer

    def get_queryset(self):
        return Subject.objects.filter(user=self.request.user).order_by('created_at')
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class  SubjectDetailView(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = SubjectSerializer
    
    lookup_field = 'id'

    def get_queryset(self):
        return Subject.objects.filter(user=self.request.user)
# ------------ documents ------------

class DocumentListCreateView(generics.ListCreateAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = DocumentSerializer

    def get_queryset(self):
        return Document.objects.filter(user=self.request.user).order_by('-created_at')

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class DocumentDetailView(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = DocumentSerializer
    lookup_field = 'id'

    def get_queryset(self):
        return Document.objects.filter(user=self.request.user)
    

# ------------- chapter ------------

class ChapterListCreateView(generics.ListCreateAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = ChapterSerializer

    def get_queryset(self):
        return Chapter.objects.filter(user=self.request.user).order_by('created_at')
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class ChapterDetailView(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = ChapterSerializer
    
    lookup_field = 'id'

    def get_queryset(self):
        return Chapter.objects.filter(user=self.request.user)
    

# ---------  chatmessage ---------------
class ChatMessageView(APIView):
    throttle_classes = [UserRateThrottle]
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):

            serializer = ChatMessageSerializer(data = request.data)

            if not serializer.is_valid():
                chat_message = serializer.save()
                logger.info(f"Chat message saved: user={request.user}, session_id={chat_message.session.id}")
                resp_data = {
                    "msg": chat_message.text,
                    "sender": chat_message.sender,
                    "created_at": chat_message.created_at,
                }
                return Response(resp_data, status=status.HTTP_201_CREATED)
            
            else:
                logger.warning(f"Invalid chat message attempt by user={request.user}: {serializer.errors}")
                return Response(
                {"error": "Invalid input", "details": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
                )
        

class ChatSessionView(generics.ListAPIView):
    permission_classes= [IsAuthenticated]
    serializer_class = ChatSessionSerializer


    def get_queryset(self):
        return ChatSession.objects.filter(user = self.request.user).order_by('-updated_at')
    
    def perform_create(self, serializer):
        serializer.save(user = self.request.user)

class ChatSessionRetriveView(generics.RetrieveAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = ChatSessionSerializer

    lookup_field = 'id'

    def get_queryset(self):
        return ChatSession.objects.filter(user=self.request.user)

        
