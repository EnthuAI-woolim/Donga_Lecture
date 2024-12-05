import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:test_project/ui/chatting/chatting_screen.dart';

class ChattingPage extends StatelessWidget {
  const ChattingPage({super.key});

  void _createChatRoom(BuildContext context) async {
    final roomNameController = TextEditingController();

    await showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: Text('새 채팅방 만들기'),
          content: TextField(
            controller: roomNameController,
            decoration: InputDecoration(hintText: '채팅방 이름 입력'),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: Text('취소'),
            ),
            TextButton(
              onPressed: () async {
                final roomName = roomNameController.text.trim();
                if (roomName.isNotEmpty) {
                  await FirebaseFirestore.instance.collection('chats').add({
                    'roomName': roomName,
                    'lastMessage': '채팅방이 생성되었습니다!',
                    'timestamp': FieldValue.serverTimestamp(),
                  });
                  Navigator.pop(context);
                }
              },
              child: Text('생성'),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('채팅 목록', style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: Colors.blue,
        actions: [
          IconButton(
            icon: Icon(Icons.add),
            onPressed: () => _createChatRoom(context),
          ),
        ],
      ),
      body: StreamBuilder<QuerySnapshot>(
        stream: FirebaseFirestore.instance
            .collection('chats')
            .orderBy('timestamp', descending: true)
            .snapshots(),
        builder: (context, snapshot) {
          if (!snapshot.hasData) {
            return Center(child: CircularProgressIndicator());
          }
          final chatRooms = snapshot.data!.docs;
          return ListView.builder(
            itemCount: chatRooms.length,
            itemBuilder: (context, index) {
              final chatRoom = chatRooms[index];
              final roomName = chatRoom['roomName'];
              final lastMessage = chatRoom['lastMessage'];
              return ListTile(
                title: Text(roomName),
                subtitle: Text(lastMessage),
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) =>
                          ChatScreen(chatRoomId: chatRoom.id),
                    ),
                  );
                },
              );
            },
          );
        },
      ),
    );
  }
}
