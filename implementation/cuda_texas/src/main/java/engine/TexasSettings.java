package engine;

import java.net.URL;

/**
 * Ta klasa laduje ustawienia dla projektui
 * @author Kacper Gorski (railman85@gmail.com)
 */
public class TexasSettings {

	public static String TEXAS_LIBRARY_PATH=null;
	private static String nazwa_biblioteki="tests_texas";
	
	public static void setTexasLibraryPath() {

		// oczytujemy sciezke z pliku texas_library_path (powinien on byc w src/main/resources)
		if (TexasSettings.TEXAS_LIBRARY_PATH==null) {
			
				URL biblioteka = TexasSettings.class.getClassLoader().getResource("libtests_texas.so");
				if (biblioteka==null)
					throw new IllegalStateException("Blad, nie zlokalizowano pliku libtests_texas.so na classpath ");
				TEXAS_LIBRARY_PATH = biblioteka.getPath();
				// ucinamy nazwe pliku, potrzebna jest tylko sciezka
				TEXAS_LIBRARY_PATH = TEXAS_LIBRARY_PATH.substring(0, TEXAS_LIBRARY_PATH.length()-17);

		}
		
		String TEXAS_LIBRARY_PATH = System.getProperty("java.library.path");

		if (! TEXAS_LIBRARY_PATH.equals(TexasSettings.TEXAS_LIBRARY_PATH) ) {
			//System.out.println("zaladowalem biblioteke CUDA z AI Texas hol'em");
			System.setProperty("java.library.path", TexasSettings.TEXAS_LIBRARY_PATH);
			System.out.println(TexasSettings.TEXAS_LIBRARY_PATH);
			System.loadLibrary(nazwa_biblioteki);
		}
		

	}
	
}
