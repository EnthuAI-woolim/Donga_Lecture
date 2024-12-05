import 'package:flutter/material.dart';

class ChattingListFrame extends StatelessWidget {
  final List<String> chatRooms = ["Room A", "Room B", "Room C"]; // 예제 데이터

  @override
  Widget build(BuildContext context) {
    return SliverList(
      delegate: SliverChildBuilderDelegate(
            (context, index) {
          return ListTile(
            title: Text(chatRooms[index]),
            onTap: () {
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text('${chatRooms[index]} 클릭됨')),
              );
            },
          );
        },
        childCount: chatRooms.length,
      ),
    );
  }
}
