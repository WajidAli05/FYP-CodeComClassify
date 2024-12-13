import java.util.Scanner;

public class Calculator {

    // Method to add two numbers
    public static int addTwoNumbers(int a, int b) {
        return a + b;
    }

    // Method to prompt for input and display the result
    public static void promptForInput() {
        Scanner scanner = new Scanner(System.in);
        
        try {
            // Prompt user for the first number
            System.out.print("Enter the first number: ");
            int firstNumber = scanner.nextInt();

            // Prompt user for the second number
            System.out.print("Enter the second number: ");
            int secondNumber = scanner.nextInt();

            // Add the numbers and display the result
            int result = addTwoNumbers(firstNumber, secondNumber);
            System.out.println("The sum of " + firstNumber + " and " + secondNumber + " is " + result);
        } catch (Exception e) {
            System.out.println("Invalid input. Please enter valid integers.");
        } finally {
            scanner.close(); // Close the scanner
        }
    }

    // Main method to run the program
    public static void main(String[] args) {
        promptForInput();
    }
}

