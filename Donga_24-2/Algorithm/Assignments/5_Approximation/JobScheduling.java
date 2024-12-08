import java.util.ArrayList;
import java.util.List;

public class JobScheduling {
    public static void main(String[] args) {
        int[] operationTime = {5, 2, 4, 3, 4, 7, 9, 2, 4, 1}; // 작업 수행 시간
        int m = 4; // 머신 수

        approxJobScheduling(operationTime, m);
    }

    public static void approxJobScheduling(int[] operationTime, int m) {
        int[] L = new int[m]; // 각 머신의 현재 종료 시간
        List<List<int[]>> schedule = new ArrayList<>(); // 각 머신의 작업 스케줄 (작업 번호, 시작 시간, 종료 시간)

        // 초기화
        for (int i = 0; i < m; i++) {
            L[i] = 0;
            schedule.add(new ArrayList<>());
        }

        // 작업 할당
        for (int i = 0; i < operationTime.length; i++) {
            int minMachine = 0;

            // 가장 빨리 끝나는 머신 찾기
            for (int j = 1; j < m; j++) {
                if (L[j] < L[minMachine]) {
                    minMachine = j;
                }
            }

            // 작업 할당
            int startTime = L[minMachine];
            int endTime = startTime + operationTime[i];
            schedule.get(minMachine).add(new int[] {i + 1, startTime, endTime}); // 작업 번호, 시작 시간, 종료 시간
            L[minMachine] = endTime;
        }

        // 스케줄 출력 (표 형식)
        printSchedule(schedule, L);
    }

    private static void printSchedule(List<List<int[]>> schedule, int[] L) {
        int maxTime = 0;
        for (int time : L) {
            if (time > maxTime) maxTime = time;
        }

        System.out.println("Job Scheduling Table");

        // 테이블 헤더 출력
        System.out.print("m\\t|");
        for (int t = 0; t <= maxTime; t++) {
            System.out.print(String.format("%4d%2s|", t, " "));
        }
        System.out.println();
    
        // 각 머신에 대해 작업을 출력
        for (int i = schedule.size() - 1; i >= 0; i--) {
            System.out.print((i + 1) + "  |");  // 머신 번호 출력
            int currentTime = 0;
    
            // 각 머신에서의 작업들 출력
            for (int[] task : schedule.get(i)) {
                int taskNumber = task[0];
                int taskStart = task[1];
                int taskEnd = task[2];
                int taskDuration = taskEnd - taskStart;
    
                // 작업이 시작되기 전까지는 빈 공간 출력
                while (currentTime < taskStart) {
                    System.out.print("     ");
                    currentTime++;
                }
    
                // 작업 출력 (시작 시 | 기호, 작업명 출력 후 | 기호)
                for (int t = 0; t < taskDuration; t++) {
                    if (t == 0) { // 첫 번째 시간에만 작업 이름 출력
                        System.out.print(String.format("%3s%-3d", "t", taskNumber));
                    } else {
                        System.out.print(String.format("%" + 7 + "s", ""));  // 작업이 진행되는 동안은 공백으로 출력
                    }
                    currentTime++;
                }
                System.out.print("|");
            }
            
            // 남은 시간 동안 빈 칸 채우기
            while (currentTime <= maxTime) {
                System.out.print("      |");
                currentTime++;
            }
            System.out.println();
        }
    }
    
}
