from codecarbon import OfflineEmissionsTracker
import logging

class CarbonAI():
    def __init__(self, country_iso_code="THA", args=None):
        logging.getLogger('apscheduler.executors.default').propagate = False
        self.tracker = OfflineEmissionsTracker(country_iso_code=country_iso_code, save_to_file=True, output_dir=args.save, log_level="error")
        self.emissions = 0
        self.power = 0

    def start(self):
        self.tracker.start()

    def stop(self):
        # Emissions as CO₂-equivalents [kg CO₂eq]
        self.emissions = round(float(self.tracker.stop()),2)  
        # Sum of cpu_energy, gpu_energy and ram_energy [W]
        self.power = round(float(self.tracker.final_emissions_data.energy_consumed)*1000, 2)