package cuda.test;

public class DaneStatystyczneUtils {

	public static double getSredniaWartosc(double[] statystyki) {
		double srednia=0.0d;
		for (int i=0; i < statystyki.length; i++) {
			srednia+=statystyki[i];
		}
		srednia/=statystyki.length;
		return srednia;
	}
	
	/**
	 * @param statystyki - tablica z danymi
	 * @return zwraca srednie odchylenie
	 */
	public static double getOdchylenie(double[] statystyki) {
		double srednia=getSredniaWartosc(statystyki);
		
		double odchylenie=0.0d;
		for (int i=0; i < statystyki.length; i++) {
			odchylenie += Math.abs( statystyki[i] - srednia );
		}
		odchylenie/=statystyki.length;
		return odchylenie;
	}
	
}
