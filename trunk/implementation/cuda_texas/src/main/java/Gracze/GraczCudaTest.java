package Gracze;

import engine.Karta;

public class GraczCudaTest extends Gracz {

	private int kolejnosc=0;
	
	public GraczCudaTest(int kolejnosc) {
		this.kolejnosc = kolejnosc;
	}
	
	@Override
	public double play(int as, double j) {
		
		Karta[] karty = gra.rozdanie.getAllCards(kolejnosc);
		int max=0;
		
		for (int i=0; i < karty.length; i++) {
			if (karty[i].wysokosc > max)
				max = karty[i].wysokosc;
		}

		if (max<6)
			return -1;
		else {
			
			if (gra.stawka > max * 10.0f) {
				bilans -= gra.stawka - j;  
				return gra.stawka;
			}
			else { 
				bilans -= max*10.0f - j;  
				return max*10.0f;
			}
		}
		
	}

}
