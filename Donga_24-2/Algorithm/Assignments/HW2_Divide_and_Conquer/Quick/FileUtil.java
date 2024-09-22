import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class FileUtil {

    public static List<Integer> readNumbersFromFile(String filename) {
        List<Integer> numbers = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                try {
                    numbers.add(Integer.parseInt(line));
                } catch (NumberFormatException e) {
                    System.err.println("숫자로 변환할 수 없는 값이 있습니다: " + line);
                }
            }
        } catch (IOException e) {
            System.err.println("파일을 열 수 없습니다!");
        }
        return numbers;
    }

    public static void writeArrayToFile(String filename, Integer[] arr) {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(filename))) {
            for (int num : arr) {
                bw.write(Integer.toString(num));
                bw.newLine();
            }
        } catch (IOException e) {
            System.err.println("파일을 열 수 없습니다!");
        }
    }
}
