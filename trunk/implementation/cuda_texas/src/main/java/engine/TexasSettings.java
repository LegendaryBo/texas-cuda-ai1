package engine;

public class TexasSettings {

	public static String TEXAS_LIBRARY_PATH="/home/railman/workspace/cuda_texas/src/main/java";
	private static String nazwa_biblioteki="tests_texas";
	
	public static void setTexasLibraryPath() {

		
		String TEXAS_LIBRARY_PATH = System.getProperty("java.library.path");

		if (! TEXAS_LIBRARY_PATH.equals(TexasSettings.TEXAS_LIBRARY_PATH) ) {
			//System.out.println("zaladowalem biblioteke CUDA z AI Texas hol'em");
			System.setProperty("java.library.path", TexasSettings.TEXAS_LIBRARY_PATH);
			System.loadLibrary(nazwa_biblioteki);
		}
		

	}
	
}
