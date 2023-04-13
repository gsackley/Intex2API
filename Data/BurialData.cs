using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Intex2API.Data
{
    public class BurialData
    {
        public float id { get; set; }
        public float square_north_south { get; set; }
        public float head_direction { get; set; }
        public float sex { get; set; }
        public float north_south { get; set; }
        public float depth { get; set; }
        public float east_west { get; set; }
        public float adult_sub_adult { get; set; }
        public float face_bundles { get; set; }
        public float south_to_head { get; set; }
        public float preservation { get; set; }
        public float fieldbook_page { get; set; }
        public float square_east_west { get; set; }
        public float goods { get; set; }
        public float text { get; set; }
        public float wrapping { get; set; }
        public float hair_color { get; set; }
        public float west_to_head { get; set; }
        public float samples_collected { get; set; }
        public float area { get; set; }
        public float burial_id { get; set; }
        public float length { get; set; }
        public float burial_number { get; set; }
        public float data_expert_initals { get; set; }
        public float west_to_feet { get; set; }
        public float age_at_death { get; set; }
        public float south_to_feet { get; set; }
        public float excavation_recorder { get; set; }
        public float photos { get; set; }
        public float hair { get; set; }
        public float burial_materials { get; set; }
        public float date_of_excavation { get; set; }
        public float fieldbook_excavation_year { get; set; }
        public float cluster_number { get; set; }
        public float shaft_number { get; set; }
        public Tensor<float> AsTensor()
        {
            float[] data = new float[]
            {
              id, square_north_south, head_direction, sex, north_south,
              depth, east_west, adult_sub_adult, face_bundles,
               south_to_head, preservation, fieldbook_page, square_east_west,
               goods, text, wrapping, hair_color, west_to_head,
               samples_collected, area, burial_id, length, burial_number,
               data_expert_initals, west_to_feet, age_at_death, south_to_feet,
               excavation_recorder, photos, hair, burial_materials,
               date_of_excavation, fieldbook_excavation_year, cluster_number,
               shaft_number
                    };
            int[] dimensions = new int[] { 1, 35 };
            return new DenseTensor<float>(data, dimensions);
        }
    }
}